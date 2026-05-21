"""MediaPipe pose estimation and hip-centered normalization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    import mediapipe as mp
except ImportError:
    mp = None  # type: ignore

NUM_LANDMARKS = 33
MIN_TORSO_SCALE = 1e-6
L_HIP, R_HIP = 23, 24
L_SHOULDER, R_SHOULDER = 11, 12


@dataclass
class PoseFrame:
    """Normalized landmarks shaped (33, 3) in body-relative space."""
    landmarks: np.ndarray        # (33, 3) normalized
    visibility: np.ndarray       # (33,) scores 0-1
    raw_landmarks: Optional[np.ndarray] = None  # (33, 3) image-space 0-1


def hip_center(landmarks: np.ndarray) -> np.ndarray:
    return (landmarks[L_HIP] + landmarks[R_HIP]) / 2.0


def torso_scale(landmarks: np.ndarray) -> float:
    left = landmarks[L_HIP, :2]
    right = landmarks[R_HIP, :2]
    return max(float(np.linalg.norm(left - right)), MIN_TORSO_SCALE)


def normalize_landmarks(raw: np.ndarray) -> np.ndarray:
    """Hip-center origin, scale by torso width."""
    out = np.asarray(raw, dtype=np.float32).copy()
    out -= hip_center(out)
    out /= torso_scale(out)
    return out


def is_pose_reliable(frame: PoseFrame, min_visibility: float = 0.5) -> bool:
    """True only if hips and shoulders are all visible above threshold."""
    key_indices = [L_HIP, R_HIP, L_SHOULDER, R_SHOULDER]
    return all(frame.visibility[i] >= min_visibility for i in key_indices)


class PoseEstimator:
    """MediaPipe Pose wrapper for live webcam/video frames."""

    def __init__(
        self,
        *,
        static_image_mode: bool = False,
        model_complexity: int = 1,
        smooth_landmarks: bool = True,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        if mp is None:
            raise ImportError("mediapipe is required. pip install mediapipe")
        self._pose = mp.solutions.pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            smooth_landmarks=smooth_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process(
        self, rgb_frame: np.ndarray, *, keep_raw: bool = False
    ) -> Optional[PoseFrame]:
        """Run pose on an RGB frame; return normalized PoseFrame or None."""
        results = self._pose.process(rgb_frame)
        if not results.pose_landmarks:
            return None
        lm = results.pose_landmarks.landmark
        raw = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)
        vis = np.array([p.visibility for p in lm], dtype=np.float32)
        normalized = normalize_landmarks(raw)
        return PoseFrame(
            landmarks=normalized,
            visibility=vis,
            raw_landmarks=raw.copy() if keep_raw else None,
        )

    def close(self) -> None:
        self._pose.close()

    def __enter__(self) -> PoseEstimator:
        return self

    def __exit__(self, *_exc) -> None:
        self.close()