"""MediaPipe pose estimation and hip-centered normalization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    import mediapipe as mp
except ImportError:  # pragma: no cover
    mp = None  # type: ignore

NUM_LANDMARKS = 33


@dataclass
class PoseFrame:
    """Normalized landmarks shaped (33, 3) as x, y, z in body-relative space."""

    landmarks: np.ndarray
    visibility: np.ndarray


def _torso_scale(landmarks: np.ndarray) -> float:
    """Distance between left and right hip (indices 23, 24 in MediaPipe Pose)."""
    left_hip = landmarks[23, :2]
    right_hip = landmarks[24, :2]
    scale = float(np.linalg.norm(left_hip - right_hip))
    return max(scale, 1e-6)


def normalize_landmarks(raw: np.ndarray) -> np.ndarray:
    """Translate to hip midpoint and scale by torso width."""
    out = raw.copy()
    hip_center = (out[23] + out[24]) / 2.0
    out -= hip_center
    scale = _torso_scale(out)
    out[:, :2] /= scale
    return out


class PoseEstimator:
    """Thin wrapper around MediaPipe Pose."""

    def __init__(self, min_detection_confidence: float = 0.5, min_tracking_confidence: float = 0.5):
        if mp is None:
            raise ImportError("mediapipe is required. Install with: pip install mediapipe")

        self._pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process(self, rgb_frame: np.ndarray) -> Optional[PoseFrame]:
        """Run pose on an RGB frame; return normalized landmarks or None."""
        results = self._pose.process(rgb_frame)
        if not results.pose_landmarks:
            return None

        lm = results.pose_landmarks.landmark
        raw = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)
        vis = np.array([p.visibility for p in lm], dtype=np.float32)
        normalized = normalize_landmarks(raw)
        return PoseFrame(landmarks=normalized, visibility=vis)

    def close(self) -> None:
        self._pose.close()
