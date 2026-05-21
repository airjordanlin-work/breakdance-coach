"""Circular buffer for reliable pose frames — 60-frame sliding window for DTW."""

from __future__ import annotations

from collections import deque

import numpy as np

from app.pose_estimator import PoseFrame, is_pose_reliable

WINDOW_LEN = 60


class PoseBuffer:
    """Fixed-size deque of PoseFrame objects for sequence comparison."""

    def __init__(self, maxlen: int = WINDOW_LEN) -> None:
        self._maxlen = maxlen
        self._frames: deque[PoseFrame] = deque(maxlen=maxlen)

    def __len__(self) -> int:
        return len(self._frames)

    def add(self, frame: PoseFrame) -> bool:
        """Append a frame if reliable; silently skip otherwise."""
        if not is_pose_reliable(frame):
            return False
        self._frames.append(frame)
        return True

    def is_ready(self) -> bool:
        """True when the buffer holds exactly maxlen reliable frames."""
        return len(self._frames) == self._maxlen

    def get_sequence(self) -> np.ndarray:
        """Stack normalized landmarks as (60, 33, 3) for the DTW engine."""
        if not self.is_ready():
            raise ValueError(
                f"Buffer not ready: {len(self._frames)}/{self._maxlen} frames"
            )
        return np.stack([f.landmarks for f in self._frames], axis=0).astype(np.float32)

    def clear(self) -> None:
        """Reset the buffer between moves."""
        self._frames.clear()

    @property
    def fill_ratio(self) -> float:
        """How full the buffer is, 0.0 to 1.0."""
        return len(self._frames) / self._maxlen