"""Output layer — pure OpenCV drawing utilities for live pose coaching HUD."""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from app.dtw_engine import DTWResult
from app.pose_estimator import PoseFrame
from app.scorer import CLOSE, MISS, PERFECT, ScoreResult, SessionStats

BODY_CONNECTIONS: list[tuple[int, int]] = [
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (11, 23),
    (12, 24),
    (23, 24),
    (23, 25),
    (25, 27),
    (24, 26),
    (26, 28),
]

KEYFRAME_FLASH_FRAMES = 30
MIN_VISIBLE_FOR_BOX = 5
BOX_VISIBILITY_MIN = 0.5
BOX_PADDING_PX = 20

COLOR_GREEN = (0, 255, 0)
COLOR_YELLOW = (0, 255, 255)
COLOR_RED = (0, 0, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_GREY = (180, 180, 180)
COLOR_DARK_GREY = (80, 80, 80)
COLOR_PANEL_BG = (17, 17, 24)
COLOR_BOX_GREY = (150, 150, 150)
COLOR_BOX_MATCHING = (0, 255, 128)

GRADE_COLORS: dict[str, tuple[int, int, int]] = {
    "S": (0, 215, 255),
    "A": (0, 255, 0),
    "B": (255, 255, 0),
    "C": (0, 255, 255),
    "D": (0, 0, 255),
}

TIER_COLORS: dict[str, tuple[int, int, int]] = {
    PERFECT: COLOR_GREEN,
    CLOSE: COLOR_YELLOW,
    MISS: COLOR_RED,
}


def _landmark_pixel(x: float, y: float, frame_w: int, frame_h: int) -> tuple[int, int]:
    """Convert MediaPipe normalized image coords to pixel (x, y)."""
    px = int(np.clip(x, 0.0, 1.0) * frame_w)
    py = int(np.clip(y, 0.0, 1.0) * frame_h)
    return px, py


def _visibility_point_color(visibility: float) -> tuple[int, int, int]:
    """Circle color from a single landmark visibility score."""
    if visibility >= 0.7:
        return COLOR_GREEN
    if visibility >= 0.4:
        return COLOR_YELLOW
    return COLOR_RED


def _visibility_line_color(visibility_a: float, visibility_b: float) -> tuple[int, int, int]:
    """Line color from average visibility of two endpoints."""
    avg = (visibility_a + visibility_b) / 2.0
    if avg >= 0.7:
        return COLOR_WHITE
    if avg >= 0.4:
        return COLOR_GREY
    return COLOR_DARK_GREY


def draw_skeleton(frame: np.ndarray, pose_frame: PoseFrame) -> np.ndarray:
    """Draw pose landmarks and body connections using raw image-space coordinates."""
    if pose_frame.raw_landmarks is None:
        return frame

    h, w = frame.shape[:2]
    raw = pose_frame.raw_landmarks
    vis = pose_frame.visibility

    points: list[tuple[int, int] | None] = []
    for i in range(raw.shape[0]):
        px, py = _landmark_pixel(float(raw[i, 0]), float(raw[i, 1]), w, h)
        points.append((px, py))
        color = _visibility_point_color(float(vis[i]))
        cv2.circle(frame, (px, py), 4, color, -1, lineType=cv2.LINE_AA)

    for a, b in BODY_CONNECTIONS:
        if a >= len(points) or b >= len(points):
            continue
        pa, pb = points[a], points[b]
        if pa is None or pb is None:
            continue
        line_color = _visibility_line_color(float(vis[a]), float(vis[b]))
        cv2.line(frame, pa, pb, line_color, 2, lineType=cv2.LINE_AA)

    return frame


def draw_bounding_box(
    frame: np.ndarray,
    pose_frame: PoseFrame,
    dtw_result: Optional[DTWResult],
) -> np.ndarray:
    """Draw alignment bounding box and status label from visible landmarks."""
    if pose_frame.raw_landmarks is None:
        return frame

    h, w = frame.shape[:2]
    raw = pose_frame.raw_landmarks
    vis = pose_frame.visibility

    xs: list[float] = []
    ys: list[float] = []
    for i in range(raw.shape[0]):
        if float(vis[i]) >= BOX_VISIBILITY_MIN:
            xs.append(float(raw[i, 0]))
            ys.append(float(raw[i, 1]))

    if len(xs) < MIN_VISIBLE_FOR_BOX:
        return frame

    x1 = int(np.clip(min(xs) * w, 0, w - 1)) - BOX_PADDING_PX
    y1 = int(np.clip(min(ys) * h, 0, h - 1)) - BOX_PADDING_PX
    x2 = int(np.clip(max(xs) * w, 0, w - 1)) + BOX_PADDING_PX
    y2 = int(np.clip(max(ys) * h, 0, h - 1)) + BOX_PADDING_PX

    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)

    if dtw_result is None:
        color = COLOR_BOX_GREY
        label = "DETECTING"
    elif dtw_result.aligned:
        color = COLOR_GREEN
        label = "ALIGNED"
    else:
        color = COLOR_BOX_MATCHING
        label = "MATCHING..."

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, lineType=cv2.LINE_AA)
    cv2.putText(
        frame,
        label,
        (x1 + 4, max(y1 - 8, 18)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )
    return frame


def draw_score_hud(frame: np.ndarray, stats: SessionStats) -> np.ndarray:
    """Draw top-left session stats panel with semi-transparent background."""
    panel_w, panel_h = 280, 150
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (10 + panel_w, 10 + panel_h), COLOR_PANEL_BG, -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

    grade_color = GRADE_COLORS.get(stats.grade, COLOR_WHITE)
    lines: list[tuple[str, tuple[int, int, int]]] = [
        (f"SCORE:  {stats.total_points:.0f}", COLOR_WHITE),
        (f"GRADE:  {stats.grade}", grade_color),
        (f"STREAK: {stats.streak}x", COLOR_WHITE),
        (f"PERFECT: {stats.perfects}  CLOSE: {stats.closes}  MISS: {stats.misses}", COLOR_GREY),
    ]

    y = 34
    for text, color in lines:
        cv2.putText(
            frame,
            text,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
        y += 28

    return frame


def draw_buffer_bar(frame: np.ndarray, fill_ratio: float) -> np.ndarray:
    """Draw bottom buffer progress bar; green fill scales with ``fill_ratio``."""
    h, w = frame.shape[:2]
    ratio = float(np.clip(fill_ratio, 0.0, 1.0))

    bar_h = 14
    margin = 12
    x1, y1 = margin, h - margin - bar_h
    x2, y2 = w - margin, h - margin

    cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_DARK_GREY, -1)

    fill_x2 = x1 + int((x2 - x1) * ratio)
    if fill_x2 > x1:
        cv2.rectangle(frame, (x1, y1), (fill_x2, y2), COLOR_GREEN, -1)

    label = "READY" if ratio >= 1.0 else "BUFFERING..."
    cv2.putText(
        frame,
        label,
        (x1 + 6, y1 - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        COLOR_WHITE,
        1,
        cv2.LINE_AA,
    )
    return frame


def draw_keyframe_flash(
    frame: np.ndarray,
    score_result: Optional[ScoreResult],
    flash_counter: int,
) -> tuple[np.ndarray, int]:
    """Flash per-joint keyframe results for up to 30 frames; return updated counter."""
    if flash_counter <= 0:
        return frame, 0
    if score_result is None:
        return frame, max(flash_counter - 1, 0)

    h, w = frame.shape[:2]
    panel_w = min(360, w - 40)
    panel_h = 24 + 22 * len(score_result.results)
    x1, y1 = w - panel_w - 20, 80
    x2, y2 = x1 + panel_w, y1 + panel_h

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), COLOR_PANEL_BG, -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    cv2.putText(
        frame,
        f"KEYFRAME +{score_result.points_this_attempt:.0f}",
        (x1 + 8, y1 + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        COLOR_WHITE,
        1,
        cv2.LINE_AA,
    )

    y = y1 + 44
    for joint in score_result.results:
        color = TIER_COLORS.get(joint.tier, COLOR_WHITE)
        text = f"{joint.joint_name}: {joint.tier} ({joint.diff:.0f}°)"
        cv2.putText(
            frame,
            text,
            (x1 + 8, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
        y += 22

    return frame, max(flash_counter - 1, 0)


def draw_move_label(frame: np.ndarray, move_name: Optional[str]) -> np.ndarray:
    """Draw current best-matching move name in the bottom-right corner."""
    if not move_name:
        return frame

    h, w = frame.shape[:2]
    text = f"MOVE: {move_name}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    x = w - tw - 16
    y = h - 36

    overlay = frame.copy()
    cv2.rectangle(overlay, (x - 8, y - th - 10), (x + tw + 8, y + 8), COLOR_PANEL_BG, -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        COLOR_WHITE,
        2,
        cv2.LINE_AA,
    )
    return frame


def render(
    frame: np.ndarray,
    *,
    pose_frame: Optional[PoseFrame] = None,
    dtw_result: Optional[DTWResult] = None,
    score_result: Optional[ScoreResult] = None,
    stats: Optional[SessionStats] = None,
    fill_ratio: float = 0.0,
    flash_counter: int = 0,
    move_name: Optional[str] = None,
) -> tuple[np.ndarray, int]:
    """Apply all overlay layers in draw order; return frame and updated flash counter."""
    out = frame

    if pose_frame is not None:
        out = draw_skeleton(out, pose_frame)
        out = draw_bounding_box(out, pose_frame, dtw_result)

    if stats is not None:
        out = draw_score_hud(out, stats)

    out = draw_buffer_bar(out, fill_ratio)
    out, updated_flash = draw_keyframe_flash(out, score_result, flash_counter)
    out = draw_move_label(out, move_name)

    return out, updated_flash
