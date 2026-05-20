"""Circular buffer for reliable pose frames — 60-frame sliding window for DTW."""

from __future__ import annotations

from collections import deque

import numpy as np

from app.pose_estimator import PoseFrame, is_pose_reliable

#we want fps at 60 because at 30fps there will be a 2-second window
#breakdancing moves r typically short thus this will make or break the program
WINDOW_LEN = 60


class PoseBuffer:
    """Fixed-size deque of :class:`PoseFrame` objects for sequence comparison."""

    def __init__(self, maxlen: int = WINDOW_LEN) -> None:
        self._maxlen = maxlen
        #deque performs append and pop at O(1) time complexity 
        #going over list 30 times per second usually list has O(n) time complexity
        self._frames: deque[PoseFrame] = deque(maxlen=maxlen)

    def __len__(self) -> int:
        return len(self._frames)

    #if pose isnt reliable we skip it
    #gate on visibility scores before frames enter buffer, so DTW comparison only 
    #sees high-confidence skeletal data
    def add(self, frame: PoseFrame) -> bool:
        """Append a frame if reliable; silently skip otherwise.

        Returns True if the frame was stored, False if skipped.
        """
        if not is_pose_reliable(frame):
            return False
        self._frames.append(frame)
        return True

    def is_ready(self) -> bool:
        """True when the buffer holds exactly ``maxlen`` reliable frames."""
        return len(self._frames) == self._maxlen

    def get_sequence(self) -> np.ndarray:
        """Stack normalized landmarks as (60, 33, 3) for the DTW engine.

        Raises:
            ValueError: If the buffer is not full yet.
        """
        if not self.is_ready():
            raise ValueError(
                f"Buffer not ready: {len(self._frames)}/{self._maxlen} frames"
            )
        return np.stack([f.landmarks for f in self._frames], axis=0).astype(np.float32)

    def clear(self) -> None:
        """Reset the buffer (e.g. between moves)."""
        self._frames.clear()
