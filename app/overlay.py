"""OpenCV HUD overlay for score, grade, and joint feedback."""

from __future__ import annotations

import cv2
import numpy as np

from app.scorer import ScoreResult


def draw_skeleton(
    frame: np.ndarray,
    landmarks: np.ndarray,
    visibility: np.ndarray,
    min_visibility: float = 0.5,
) -> None:
    """Draw a simple stick figure from normalized landmarks mapped to frame size."""
    h, w = frame.shape[:2]
    points = []
    for i, (x, y, _z) in enumerate(landmarks):
        if visibility[i] < min_visibility:
            points.append(None)
            continue
        px = int((x + 0.5) * w)
        py = int((y + 0.5) * h)
        points.append((px, py))
        cv2.circle(frame, (px, py), 3, (124, 108, 252), -1)

    connections = [
        (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
        (11, 23), (12, 24), (23, 24), (23, 25), (25, 27), (24, 26), (26, 28),
    ]
    for a, b in connections:
        if points[a] and points[b]:
            cv2.line(frame, points[a], points[b], (252, 108, 156), 2)


def draw_hud(
    frame: np.ndarray,
    *,
    score: int,
    grade: str,
    move_name: str | None,
    dtw_distance: float | None,
    last_result: ScoreResult | None,
    dtw_aligned: bool | None = None,
) -> None:
    """Render score panel and correction hints."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (320, 140), (17, 17, 24), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    lines = [
        f"Score: {score}",
        f"Grade: {grade}",
        f"Move: {move_name or '—'}",
    ]
    if dtw_distance is not None:
        status = ""
        if dtw_aligned is True:
            status = " ✓"
        elif dtw_aligned is False:
            status = " ✗"
        lines.append(f"DTW: {dtw_distance:.1f}{status}")

    y = 36
    for line in lines:
        cv2.putText(
            frame, line, (20, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (226, 226, 240), 1, cv2.LINE_AA,
        )
        y += 26

    if last_result and last_result.feedback:
        hint = last_result.feedback[0][:48]
        cv2.putText(
            frame, hint, (20, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (252, 108, 156), 1, cv2.LINE_AA,
        )

    if last_result and last_result.hit:
        cv2.putText(
            frame, f"COMBO x{last_result.combo_multiplier}!", (20, 170),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (124, 252, 180), 2, cv2.LINE_AA,
        )
