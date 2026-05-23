"""Output layer — polished OpenCV drawing utilities for live pose coaching HUD."""

from __future__ import annotations

from typing import Optional
import math

import cv2
import numpy as np

from app.dtw_engine import DTWResult
from app.pose_estimator import PoseFrame, is_pose_reliable
from app.scorer import CLOSE, MISS, PERFECT, ScoreResult, SessionStats

BODY_CONNECTIONS: list[tuple[int, int]] = [
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (11, 23), (12, 24),
    (23, 24),
    (23, 25), (25, 27),
    (24, 26), (26, 28),
]

# Ghost T-pose — tightly proportioned, fits within center 40% of frame width
# x values kept between 0.30 and 0.70 so arms don't span the whole screen
# T-pose ghost in BODY SPACE — units relative to hip-center, scaled by shoulder width
GHOST_BONES_BODY = [
    ( 0.00, -1.80,  0.00, -1.50),   # neck
    (-0.50, -1.45,  0.50, -1.45),   # shoulders
    (-0.50, -1.45, -1.10, -1.45),   # left upper arm
    (-1.10, -1.45, -1.65, -1.45),   # left forearm
    ( 0.50, -1.45,  1.10, -1.45),   # right upper arm
    ( 1.10, -1.45,  1.65, -1.45),   # right forearm
    (-0.50, -1.45, -0.20, -0.80),   # left torso
    ( 0.50, -1.45,  0.20, -0.80),   # right torso
    (-0.20, -0.80,  0.20, -0.80),   # hips
    (-0.20, -0.80, -0.22, -0.20),   # left thigh
    ( 0.20, -0.80,  0.22, -0.20),   # right thigh
    (-0.22, -0.20, -0.22,  0.48),   # left shin
    ( 0.22, -0.20,  0.22,  0.48),   # right shin
]

GHOST_JOINTS_BODY = [
    ( 0.00, -1.90),
    (-0.50, -1.45), ( 0.50, -1.45),
    (-1.10, -1.45), ( 1.10, -1.45),
    (-1.65, -1.45), ( 1.65, -1.45),
    (-0.20, -0.80), ( 0.20, -0.80),
    (-0.22, -0.20), ( 0.22, -0.20),
    (-0.22,  0.48), ( 0.22,  0.48),
]

def _body_to_pixel(
    bx: float, by: float,
    hip_cx: int, hip_cy: int,
    scale: float,
) -> tuple[int, int]:
    """Convert body-space coords to pixel coords using hip anchor and scale."""
    px = int(hip_cx + bx * scale)
    py = int(hip_cy + by * scale)
    return px, py


KEYFRAME_FLASH_FRAMES = 30

COLOR_GREEN     = (0, 220, 0)
COLOR_YELLOW    = (0, 210, 210)
COLOR_RED       = (50, 50, 220)
COLOR_WHITE     = (240, 240, 240)
COLOR_GREY      = (160, 160, 160)
COLOR_DARK_GREY = (60, 60, 60)
COLOR_PANEL_BG  = (12, 12, 18)
COLOR_BORDER    = (55, 55, 85)

GRADE_COLORS: dict[str, tuple[int, int, int]] = {
    "S": (0, 200, 255),
    "A": (0, 220, 0),
    "B": (0, 220, 220),
    "C": (0, 200, 255),
    "D": (50, 50, 220),
}

TIER_COLORS: dict[str, tuple[int, int, int]] = {
    PERFECT: (0, 220, 0),
    CLOSE:   (0, 200, 200),
    MISS:    (50, 50, 220),
}

L_HIP, R_HIP           = 23, 24
L_SHOULDER, R_SHOULDER = 11, 12

FONT = cv2.FONT_HERSHEY_SIMPLEX


def _px(x: float, y: float, w: int, h: int) -> tuple[int, int]:
    return int(np.clip(x, 0.0, 1.0) * w), int(np.clip(y, 0.0, 1.0) * h)


def _panel(
    frame: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    alpha: float = 0.82,
    radius: int = 8,
) -> None:
    """Draw a semi-transparent rounded panel."""
    overlay = frame.copy()
    # rounded rect approximation
    cv2.rectangle(overlay, (x1 + radius, y1), (x2 - radius, y2), COLOR_PANEL_BG, -1)
    cv2.rectangle(overlay, (x1, y1 + radius), (x2, y2 - radius), COLOR_PANEL_BG, -1)
    for cx, cy in [(x1+radius, y1+radius), (x2-radius, y1+radius),
                   (x1+radius, y2-radius), (x2-radius, y2-radius)]:
        cv2.circle(overlay, (cx, cy), radius, COLOR_PANEL_BG, -1)
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)
    # border
    cv2.rectangle(frame, (x1 + radius, y1), (x2 - radius, y2), COLOR_BORDER, 1)
    cv2.rectangle(frame, (x1, y1 + radius), (x2, y2 - radius), COLOR_BORDER, 1)


def _text(frame, text, x, y, color=COLOR_WHITE, scale=0.48, thick=1):
    cv2.putText(frame, text, (x, y), FONT, scale, (0,0,0), thick+2, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), FONT, scale, color, thick, cv2.LINE_AA)


def _dashed_line(img, p1, p2, color, thickness=2, dash=8, gap=5):
    x1, y1 = p1; x2, y2 = p2
    length = math.hypot(x2-x1, y2-y1)
    if length == 0: return
    ux, uy = (x2-x1)/length, (y2-y1)/length
    pos, draw = 0.0, True
    while pos < length:
        seg = dash if draw else gap
        end = min(pos+seg, length)
        if draw:
            cv2.line(img,
                     (int(x1+ux*pos), int(y1+uy*pos)),
                     (int(x1+ux*end), int(y1+uy*end)),
                     color, thickness, cv2.LINE_AA)
        pos = end; draw = not draw


# ── skeleton ──────────────────────────────────────────────────────────────────

def draw_skeleton(frame: np.ndarray, pose_frame: PoseFrame) -> np.ndarray:
    if pose_frame.raw_landmarks is None:
        return frame
    h, w = frame.shape[:2]
    raw, vis = pose_frame.raw_landmarks, pose_frame.visibility
    pts = []
    for i in range(raw.shape[0]):
        px, py = _px(float(raw[i,0]), float(raw[i,1]), w, h)
        pts.append((px, py))
        v = float(vis[i])
        c = (0,200,0) if v>=0.7 else ((0,180,180) if v>=0.4 else (50,50,200))
        cv2.circle(frame, (px,py), 5, (0,0,0), -1, cv2.LINE_AA)
        cv2.circle(frame, (px,py), 4, c, -1, cv2.LINE_AA)

    for a, b in BODY_CONNECTIONS:
        if a >= len(pts) or b >= len(pts): continue
        va, vb = float(vis[a]), float(vis[b])
        avg = (va+vb)/2
        lc = (200,200,200) if avg>=0.7 else ((120,120,120) if avg>=0.4 else (60,60,60))
        cv2.line(frame, pts[a], pts[b], lc, 2, cv2.LINE_AA)
    return frame


# ── ghost reference ───────────────────────────────────────────────────────────
def draw_reference_ghost(
    frame,
    dtw_result,
    buf_len: int,
    pose_frame=None,      # NEW — pass the live PoseFrame for anchoring
) -> object:
    """Draw ghost T-pose anchored to the user's detected body position.
 
    Falls back to center-screen positioning if no pose is detected.
    """
    import cv2
    import numpy as np
    from typing import Optional
 
    if buf_len < 60:
        return frame
 
    h, w = frame.shape[:2]
    aligned = dtw_result is not None and dtw_result.aligned
 
    bone_color  = (0, 220, 60)  if aligned else (40, 170, 170)
    joint_fill  = (0, 240, 80)  if aligned else (50, 190, 190)
    alpha_bones = 0.55          if aligned else 0.30
    alpha_joints= 0.70          if aligned else 0.45
 
    # ── compute anchor from live landmarks ───────────────────────────────────
    hip_cx, hip_cy = w // 2, int(h * 0.60)   # fallback center
    scale = w * 0.12                           # fallback scale
 
    if pose_frame is not None and pose_frame.raw_landmarks is not None:
        raw = pose_frame.raw_landmarks         # (33, 3) image-space 0-1
        vis = pose_frame.visibility
 
        L_SHOULDER, R_SHOULDER = 11, 12
        L_HIP, R_HIP           = 23, 24
 
        ls = raw[L_SHOULDER]; rs = raw[R_SHOULDER]
        lh = raw[L_HIP];      rh = raw[R_HIP]
 
        # only anchor if all four landmarks are reasonably visible
        if all(vis[i] >= 0.4 for i in [L_SHOULDER, R_SHOULDER, L_HIP, R_HIP]):
            # hip midpoint in pixels
            hip_cx = int(((lh[0] + rh[0]) / 2) * w)
            hip_cy = int(((lh[1] + rh[1]) / 2) * h)
 
            # scale = shoulder width in pixels
            shoulder_px = abs(rs[0] - ls[0]) * w
            scale = max(shoulder_px * 0.9, w * 0.06)  # floor at 6% frame width
 
    # ── draw bones ────────────────────────────────────────────────────────────
    ghost = frame.copy()
 
    def _dashed(img, p1, p2, color, thick=2, dash=9, gap=5):
        import math
        x1,y1 = p1; x2,y2 = p2
        length = math.hypot(x2-x1, y2-y1)
        if length == 0: return
        ux, uy = (x2-x1)/length, (y2-y1)/length
        pos, draw = 0.0, True
        while pos < length:
            seg = dash if draw else gap
            end = min(pos+seg, length)
            if draw:
                cv2.line(img,
                         (int(x1+ux*pos), int(y1+uy*pos)),
                         (int(x1+ux*end), int(y1+uy*end)),
                         color, thick, cv2.LINE_AA)
            pos = end; draw = not draw
 
    for bx1, by1, bx2, by2 in GHOST_BONES_BODY:
        p1 = _body_to_pixel(bx1, by1, hip_cx, hip_cy, scale)
        p2 = _body_to_pixel(bx2, by2, hip_cx, hip_cy, scale)
        # clip to frame bounds
        p1 = (max(0,min(w-1,p1[0])), max(0,min(h-1,p1[1])))
        p2 = (max(0,min(w-1,p2[0])), max(0,min(h-1,p2[1])))
        if aligned:
            cv2.line(ghost, p1, p2, bone_color, 2, cv2.LINE_AA)
        else:
            _dashed(ghost, p1, p2, bone_color)
 
    cv2.addWeighted(ghost, alpha_bones, frame, 1.0-alpha_bones, 0, frame)
 
    # ── draw joints ───────────────────────────────────────────────────────────
    jlayer = frame.copy()
 
    # head circle
    hpx = _body_to_pixel(0.00, -1.90, hip_cx, hip_cy, scale)
    head_r = max(int(scale * 0.28), 8)
    cv2.circle(jlayer, hpx, head_r+2, (0,0,0), -1, cv2.LINE_AA)
    cv2.circle(jlayer, hpx, head_r,
               bone_color if aligned else (40,170,170),
               -1 if aligned else 2, cv2.LINE_AA)
 
    for jx, jy in GHOST_JOINTS_BODY[1:]:   # skip head
        px = _body_to_pixel(jx, jy, hip_cx, hip_cy, scale)
        px = (max(0,min(w-1,px[0])), max(0,min(h-1,px[1])))
        cv2.circle(jlayer, px, 7, (0,0,0), -1, cv2.LINE_AA)
        cv2.circle(jlayer, px, 4, joint_fill, -1, cv2.LINE_AA)
 
    cv2.addWeighted(jlayer, alpha_joints, frame, 1.0-alpha_joints, 0, frame)
 
    # ── status pill ───────────────────────────────────────────────────────────
    label       = "ALIGNED" if aligned else "TARGET POSE"
    label_color = (0, 240, 80) if aligned else (40, 200, 200)
    font        = cv2.FONT_HERSHEY_SIMPLEX
    lscale      = 0.48
    (tw, th), _ = cv2.getTextSize(label, font, lscale, 1)
    lx = w//2 - tw//2
    ly = 18
    px1, py1 = lx-10, ly-th-6
    px2, py2 = lx+tw+10, ly+6
 
    overlay = frame.copy()
    cv2.rectangle(overlay, (px1,py1), (px2,py2), (10,10,20), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    cv2.rectangle(frame, (px1,py1), (px2,py2), label_color, 1, cv2.LINE_AA)
 
    cv2.putText(frame, label, (lx,ly), font, lscale, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(frame, label, (lx,ly), font, lscale, label_color, 1, cv2.LINE_AA)
 
    return frame
 


# ── body warning ──────────────────────────────────────────────────────────────

def draw_body_warning(
    frame: np.ndarray,
    pose_frame: Optional[PoseFrame],
    buf_len: int,
) -> np.ndarray:
    h, w = frame.shape[:2]

    if pose_frame is None or not is_pose_reliable(pose_frame):
        msg   = "FULL BODY REQUIRED"
        sub   = "Step back until hips and shoulders are visible"
        (tw,th),_ = cv2.getTextSize(msg, FONT, 0.65, 2)
        (sw,_),_  = cv2.getTextSize(sub, FONT, 0.40, 1)
        pw = max(tw, sw) + 36
        ph = 72
        x1 = w//2 - pw//2; y1 = h//2 - ph//2
        _panel(frame, x1, y1, x1+pw, y1+ph, alpha=0.85)
        cv2.rectangle(frame, (x1,y1), (x1+pw,y1+ph), (50,50,180), 1, cv2.LINE_AA)
        _text(frame, msg, x1+18, y1+28, (80,80,230), 0.62, 2)
        _text(frame, sub, x1+18, y1+56, COLOR_GREY, 0.38, 1)
        return frame

    return frame


# ── position guide ────────────────────────────────────────────────────────────

def draw_position_guide(
    frame: np.ndarray,
    pose_frame: Optional[PoseFrame],
    buf_len: int,
) -> np.ndarray:
    """Minimal centered silhouette — fades as buffer fills."""
    if buf_len >= 60:
        return frame

    h, w  = frame.shape[:2]
    cx    = w // 2
    fade  = max(0.0, 1.0 - buf_len / 60.0)
    c     = int(110 * fade)
    color = (c, c, c)
    thick = 1

    hr  = max(int(h * 0.038), 10)
    hcy = int(h * 0.13)
    shy = hcy + hr + int(h*0.05)
    sw  = int(w * 0.10)
    aex = int(w * 0.18)
    by  = shy + int(h * 0.22)
    bw  = int(w * 0.06)
    ky  = by  + int(h * 0.15)
    ay  = ky  + int(h * 0.15)
    ko  = int(w * 0.03)

    guide = frame.copy()
    cv2.circle(guide, (cx, hcy), hr, color, thick, cv2.LINE_AA)
    cv2.line(guide, (cx, hcy+hr), (cx, shy), color, thick, cv2.LINE_AA)
    cv2.line(guide, (cx-sw, shy), (cx+sw, shy), color, thick, cv2.LINE_AA)
    cv2.line(guide, (cx-sw, shy), (cx-aex, shy), color, thick, cv2.LINE_AA)
    cv2.line(guide, (cx+sw, shy), (cx+aex, shy), color, thick, cv2.LINE_AA)
    cv2.line(guide, (cx-sw, shy), (cx-bw, by), color, thick, cv2.LINE_AA)
    cv2.line(guide, (cx+sw, shy), (cx+bw, by), color, thick, cv2.LINE_AA)
    cv2.line(guide, (cx-bw, by), (cx+bw, by), color, thick, cv2.LINE_AA)
    cv2.line(guide, (cx-bw, by), (cx-ko, ky), color, thick, cv2.LINE_AA)
    cv2.line(guide, (cx-ko, ky), (cx-ko, ay), color, thick, cv2.LINE_AA)
    cv2.line(guide, (cx+bw, by), (cx+ko, ky), color, thick, cv2.LINE_AA)
    cv2.line(guide, (cx+ko, ky), (cx+ko, ay), color, thick, cv2.LINE_AA)

    alpha = 0.15 + 0.20 * fade
    cv2.addWeighted(guide, alpha, frame, 1.0-alpha, 0, frame)

    _text(frame, "STAND HERE",
          cx - 44, hcy - hr - 10, COLOR_WHITE, 0.44, 1)
    _text(frame, "Arms straight out  —  T-pose",
          cx - 100, ay + 20, COLOR_GREY, 0.36, 1)
    return frame


# ── distance indicator ────────────────────────────────────────────────────────

def draw_distance_indicator(frame, pose_frame):
    if pose_frame is None: return frame
    h, w = frame.shape[:2]
    vis  = pose_frame.visibility
    hips_ok = vis[L_HIP]>=0.5 and vis[R_HIP]>=0.5
    sh_ok   = vis[L_SHOULDER]>=0.5 and vis[R_SHOULDER]>=0.5
    if not hips_ok and sh_ok:
        _draw_arrow(frame, (w//2, h-28), "down", "Step back", (0,200,200))
    elif not sh_ok and hips_ok:
        _draw_arrow(frame, (w//2, 28), "up", "Step forward", (0,200,200))
    return frame


def _draw_arrow(frame, tip, direction, label, color):
    x, y = tip; s = 16
    if direction == "down":
        pts = np.array([[x,y],[x-s,y-s*2],[x+s,y-s*2]])
        lp  = (x-46, y-s*2-8)
    else:
        pts = np.array([[x,y],[x-s,y+s*2],[x+s,y+s*2]])
        lp  = (x-46, y+s*2+20)
    cv2.fillPoly(frame, [pts], color)
    _text(frame, label, lp[0], lp[1], color, 0.45, 1)


# ── score HUD ─────────────────────────────────────────────────────────────────

def draw_score_hud(frame: np.ndarray, stats: SessionStats) -> np.ndarray:
    pw, ph = 220, 130
    x1, y1 = 14, 14
    _panel(frame, x1, y1, x1+pw, y1+ph, alpha=0.85)

    gc = GRADE_COLORS.get(stats.grade, COLOR_WHITE)

    # score large
    score_str = f"{int(stats.total_points):,}"
    _text(frame, score_str, x1+14, y1+38, COLOR_WHITE, 0.80, 2)

    # grade badge
    (gw, gh), _ = cv2.getTextSize(stats.grade, FONT, 0.70, 2)
    gx, gy = x1+pw-gw-18, y1+40
    cv2.circle(frame, (gx+gw//2, gy-gh//2), max(gw,gh)//2+8, gc, 1, cv2.LINE_AA)
    _text(frame, stats.grade, gx, gy, gc, 0.70, 2)

    # divider
    cv2.line(frame, (x1+10, y1+50), (x1+pw-10, y1+50), COLOR_BORDER, 1)

    # stats row
    _text(frame, f"STREAK  {stats.streak}x", x1+14, y1+72, COLOR_GREY, 0.38, 1)
    _text(frame, f"P {stats.perfects}", x1+14,  y1+92, (0,200,0),   0.38, 1)
    _text(frame, f"C {stats.closes}",  x1+60,  y1+92, (0,190,190), 0.38, 1)
    _text(frame, f"M {stats.misses}",  x1+104, y1+92, (80,80,210), 0.38, 1)

    # progress bar inside panel
    bar_x1, bar_y1 = x1+10, y1+105
    bar_x2, bar_y2 = x1+pw-10, y1+118
    total = max(stats.perfects + stats.closes + stats.misses, 1)
    p_w = int((bar_x2-bar_x1) * stats.perfects / total)
    c_w = int((bar_x2-bar_x1) * stats.closes   / total)
    cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x2, bar_y2), (30,30,40), -1)
    if p_w: cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x1+p_w, bar_y2), (0,180,0), -1)
    if c_w: cv2.rectangle(frame, (bar_x1+p_w, bar_y1), (bar_x1+p_w+c_w, bar_y2), (0,160,160), -1)
    cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x2, bar_y2), COLOR_BORDER, 1)

    return frame


# ── buffer bar ────────────────────────────────────────────────────────────────

def draw_buffer_bar(frame: np.ndarray, fill_ratio: float) -> np.ndarray:
    h, w   = frame.shape[:2]
    ratio  = float(np.clip(fill_ratio, 0.0, 1.0))
    bh, mg = 6, 14
    x1, y1 = mg, h - mg - bh
    x2, y2 = w - mg, h - mg

    cv2.rectangle(frame, (x1,y1), (x2,y2), (25,25,25), -1)
    fx2 = x1 + int((x2-x1)*ratio)
    if fx2 > x1:
        fc = (0,200,0) if ratio>=1.0 else (0,160,200)
        cv2.rectangle(frame, (x1,y1), (fx2,y2), fc, -1)
    cv2.rectangle(frame, (x1,y1), (x2,y2), (50,50,50), 1)

    if ratio < 1.0:
        pct = f"{int(ratio*100)}%"
        _text(frame, pct, x1+6, y1-5, (0,160,200), 0.35, 1)
    else:
        _text(frame, "READY", x1+6, y1-5, (0,200,0), 0.35, 1)
    return frame


# ── keyframe flash ────────────────────────────────────────────────────────────

def draw_keyframe_flash(
    frame: np.ndarray,
    score_result: Optional[ScoreResult],
    flash_counter: int,
) -> tuple[np.ndarray, int]:
    if flash_counter <= 0:
        return frame, 0
    if score_result is None:
        return frame, max(flash_counter-1, 0)

    h, w   = frame.shape[:2]
    pw     = 260
    row_h  = 36
    ph     = 48 + row_h * len(score_result.results)
    x1     = w - pw - 14
    y1     = 60
    _panel(frame, x1, y1, x1+pw, y1+ph, alpha=0.88)

    # points header
    pts_str = f"+{score_result.points_this_attempt:.0f} pts"
    _text(frame, pts_str, x1+14, y1+24, COLOR_WHITE, 0.58, 2)

    # tier badge
    overall = score_result.results[0].tier if score_result.results else ""
    oc = TIER_COLORS.get(overall, COLOR_WHITE)
    _text(frame, overall.upper(), x1+pw-80, y1+24, oc, 0.44, 1)

    cv2.line(frame, (x1+10, y1+32), (x1+pw-10, y1+32), COLOR_BORDER, 1)

    y = y1 + 48
    for jr in score_result.results:
        jc   = TIER_COLORS.get(jr.tier, COLOR_WHITE)
        name = jr.joint_name.replace("_", " ")
        diff = jr.diff

        # accuracy bar
        bar_w = pw - 28
        fill  = int(bar_w * max(0.0, 1.0 - diff/90.0))
        cv2.rectangle(frame, (x1+14, y-8), (x1+14+bar_w, y-2), (25,25,35), -1)
        if fill > 0:
            cv2.rectangle(frame, (x1+14, y-8), (x1+14+fill, y-2), jc, -1)

        _text(frame, f"{name}", x1+14, y+12, jc, 0.38, 1)
        _text(frame, f"{diff:.0f} deg  {jr.tier}", x1+pw-110, y+12, jc, 0.36, 1)
        y += row_h

    return frame, max(flash_counter-1, 0)


# ── move label ────────────────────────────────────────────────────────────────

def draw_move_label(frame: np.ndarray, move_name: Optional[str]) -> np.ndarray:
    if not move_name:
        return frame
    h, w = frame.shape[:2]
    label = move_name.replace("_", " ").upper()
    (tw, th), _ = cv2.getTextSize(label, FONT, 0.40, 1)
    x1 = w - tw - 28; y1 = h - 36
    _panel(frame, x1-8, y1-th-8, x1+tw+8, y1+8, alpha=0.75, radius=6)
    _text(frame, label, x1, y1, COLOR_GREY, 0.40, 1)
    return frame


# ── render ────────────────────────────────────────────────────────────────────

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
    buf_len: int = 0,
) -> tuple[np.ndarray, int]:
    out = frame
    out = draw_position_guide(out, pose_frame, buf_len)
    out = draw_reference_ghost(out, dtw_result, buf_len, pose_frame)
    if pose_frame is not None:
        out = draw_skeleton(out, pose_frame)
        out = draw_distance_indicator(out, pose_frame)
    out = draw_body_warning(out, pose_frame, buf_len)
    if stats is not None:
        out = draw_score_hud(out, stats)
    out = draw_buffer_bar(out, fill_ratio)
    out, updated_flash = draw_keyframe_flash(out, score_result, flash_counter)
    out = draw_move_label(out, move_name)
    return out, updated_flash