"""Tests for PoseBuffer — run with: pytest tests/test_buffer.py"""

import numpy as np
import pytest
from app.buffer import PoseBuffer, WINDOW_LEN
from app.pose_estimator import PoseFrame


def make_reliable_frame() -> PoseFrame:
    """Helper — creates a fake PoseFrame that passes is_pose_reliable()."""
    landmarks = np.random.rand(33, 3).astype(np.float32)
    visibility = np.ones(33, dtype=np.float32)  # all landmarks fully visible
    return PoseFrame(landmarks=landmarks, visibility=visibility)


def make_unreliable_frame() -> PoseFrame:
    """Helper — creates a fake PoseFrame that fails is_pose_reliable()."""
    landmarks = np.random.rand(33, 3).astype(np.float32)
    visibility = np.zeros(33, dtype=np.float32)  # all landmarks invisible
    return PoseFrame(landmarks=landmarks, visibility=visibility)


def test_buffer_starts_empty():
    buf = PoseBuffer()
    assert len(buf) == 0
    assert not buf.is_ready()


def test_reliable_frame_is_added():
    buf = PoseBuffer()
    added = buf.add(make_reliable_frame())
    assert added is True
    assert len(buf) == 1


def test_unreliable_frame_is_skipped():
    buf = PoseBuffer()
    added = buf.add(make_unreliable_frame())
    assert added is False
    assert len(buf) == 0


def test_buffer_fills_to_maxlen():
    buf = PoseBuffer()
    for _ in range(WINDOW_LEN):
        buf.add(make_reliable_frame())
    assert buf.is_ready()
    assert len(buf) == WINDOW_LEN


def test_buffer_does_not_exceed_maxlen():
    buf = PoseBuffer()
    for _ in range(WINDOW_LEN + 10):  # overfill on purpose
        buf.add(make_reliable_frame())
    assert len(buf) == WINDOW_LEN  # deque drops oldest, never exceeds


def test_get_sequence_shape():
    buf = PoseBuffer()
    for _ in range(WINDOW_LEN):
        buf.add(make_reliable_frame())
    seq = buf.get_sequence()
    assert seq.shape == (WINDOW_LEN, 33, 3)
    assert seq.dtype == np.float32


def test_get_sequence_raises_when_not_ready():
    buf = PoseBuffer()
    with pytest.raises(ValueError):
        buf.get_sequence()


def test_clear_resets_buffer():
    buf = PoseBuffer()
    for _ in range(WINDOW_LEN):
        buf.add(make_reliable_frame())
    assert buf.is_ready()
    buf.clear()
    assert len(buf) == 0
    assert not buf.is_ready()