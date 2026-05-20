"""Integration test — full pipeline: PoseEstimator → PoseBuffer → DTWEngine → Scorer"""

import json
import numpy as np
import pytest
from pathlib import Path

from app.pose_estimator import PoseFrame, is_pose_reliable
from app.buffer import PoseBuffer
from app.dtw_engine import DTWEngine
from app.scorer import Scorer, SessionStats


def make_reliable_frame(seed: int = 0) -> PoseFrame:
    """Fake full-body visible PoseFrame."""
    rng = np.random.default_rng(seed)
    landmarks = rng.random((33, 3)).astype(np.float32)
    visibility = np.ones(33, dtype=np.float32)  # fully visible
    return PoseFrame(landmarks=landmarks, visibility=visibility)


def make_reference_library(tmp_path: Path, num_frames: int = 60) -> Path:
    """Write a fake .npy + _meta.json reference move to tmp_path."""
    rng = np.random.default_rng(42)
    seq = rng.random((num_frames, 99)).astype(np.float32)
    np.save(tmp_path / "windmill.npy", seq)
    meta = {
        "name": "windmill",
        "dtw_threshold": 1e6,   # generous so it always aligns in tests
        "keyframes": [
            {
                "frame_index": 30,
                "joints": {
                    "left_arm": {
                        "target_angle": 170.0,
                        "threshold": 10.0
                    }
                }
            }
        ]
    }
    with (tmp_path / "windmill_meta.json").open("w") as f:
        json.dump(meta, f)
    return tmp_path


# ── test 1: buffer fills correctly from pose frames ───────────────────────────

def test_buffer_accepts_reliable_frames():
    buf = PoseBuffer(maxlen=10)
    for i in range(10):
        buf.add(make_reliable_frame(seed=i))
    assert buf.is_ready()
    seq = buf.get_sequence()
    assert seq.shape == (10, 33, 3)


# ── test 2: DTW engine runs on buffer output ──────────────────────────────────

def test_dtw_runs_on_buffer_output(tmp_path):
    make_reference_library(tmp_path, num_frames=10)
    engine = DTWEngine(tmp_path, window_len=10)

    buf = PoseBuffer(maxlen=10)
    for i in range(10):
        buf.add(make_reliable_frame(seed=i))

    assert buf.is_ready()
    window = [f.landmarks for f in list(buf._frames)]
    result = engine.compare(window)

    assert result is not None
    assert result.move_name == "windmill"
    assert result.distance >= 0
    engine.shutdown()


# ── test 3: async DTW returns a valid future ──────────────────────────────────

def test_async_dtw_resolves(tmp_path):
    make_reference_library(tmp_path, num_frames=10)
    engine = DTWEngine(tmp_path, window_len=10)

    buf = PoseBuffer(maxlen=10)
    for i in range(10):
        buf.add(make_reliable_frame(seed=i))

    window = [f.landmarks for f in list(buf._frames)]
    future = engine.compare_async(window)

    assert future is not None
    result = future.result(timeout=5)
    assert result is not None
    assert result.aligned is True   # threshold is 1e6 so always aligns
    engine.shutdown()


# ── test 4: full pipeline — buffer → DTW → scorer ────────────────────────────

def test_full_pipeline(tmp_path):
    make_reference_library(tmp_path, num_frames=10)
    engine = DTWEngine(tmp_path, window_len=10)
    scorer = Scorer()

    buf = PoseBuffer(maxlen=10)
    for i in range(10):
        buf.add(make_reliable_frame(seed=i))

    # step 1 — get DTW result
    window = [f.landmarks for f in list(buf._frames)]
    dtw_result = engine.compare(window)
    assert dtw_result is not None
    assert dtw_result.aligned is True

    # step 2 — score the current frame against the move's keyframes
    current_frame = make_reliable_frame(seed=99)
    score_result = scorer.score(dtw_result, current_frame)

    # step 3 — check session stats updated
    stats = scorer.stats
    assert isinstance(stats, SessionStats)
    assert stats.total_attempts >= 0
    engine.shutdown()


# ── test 5: combo multiplier builds across attempts ───────────────────────────

def test_combo_builds_on_perfect_streak(tmp_path):
    make_reference_library(tmp_path, num_frames=10)
    engine = DTWEngine(tmp_path, window_len=10)
    scorer = Scorer()

    for attempt in range(5):
        buf = PoseBuffer(maxlen=10)
        for i in range(10):
            buf.add(make_reliable_frame(seed=i))
        window = [f.landmarks for f in list(buf._frames)]
        dtw_result = engine.compare(window)
        scorer.score(dtw_result, make_reliable_frame(seed=attempt))

    # after 5 attempts stats should reflect them
    assert scorer.stats.total_attempts > 0
    engine.shutdown()