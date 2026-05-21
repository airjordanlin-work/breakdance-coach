"""Integration test — full pipeline: PoseEstimator → PoseBuffer → DTWEngine → Scorer"""

import json

import numpy as np
import pytest
from pathlib import Path

from app.pose_estimator import PoseFrame, is_pose_reliable
from app.buffer import PoseBuffer
from app.dtw_engine import DTWEngine, DTWResult
from app.scorer import Scorer, SessionStats


def make_reliable_frame(seed: int = 0) -> PoseFrame:
    """Fake full-body visible PoseFrame."""
    rng = np.random.default_rng(seed)
    landmarks = rng.random((33, 3)).astype(np.float32)
    visibility = np.ones(33, dtype=np.float32)
    return PoseFrame(landmarks=landmarks, visibility=visibility)


def make_reference_library(tmp_path: Path, num_frames: int = 10) -> Path:
    """Write a fake .npy + _meta.json reference move to tmp_path."""
    rng = np.random.default_rng(42)
    seq = rng.random((num_frames, 99)).astype(np.float32)
    np.save(tmp_path / "windmill.npy", seq)
    meta = {
        "name": "windmill",
        "dtw_threshold": 1e6,
        "keyframes": [
            {
                "frame": 5,
                "joints": [
                    {
                        "joint_name": "left_elbow",
                        "joint_triplet": [11, 13, 15],
                        "target_angle": 90.0,
                        "threshold": 90.0,
                    }
                ],
            }
        ],
    }
    with (tmp_path / "windmill_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f)
    return tmp_path


def make_scorable_pose() -> PoseFrame:
    """PoseFrame that should score Perfect against the test keyframe."""
    landmarks = np.zeros((33, 3), dtype=np.float32)
    landmarks[11] = [1.0, 0.0, 0.0]
    landmarks[13] = [0.0, 0.0, 0.0]
    landmarks[15] = [0.0, 1.0, 0.0]
    visibility = np.ones(33, dtype=np.float32)
    return PoseFrame(landmarks=landmarks, visibility=visibility)


def test_buffer_accepts_reliable_frames() -> None:
    buf = PoseBuffer(maxlen=10)
    for i in range(10):
        buf.add(make_reliable_frame(seed=i))
    assert buf.is_ready()
    seq = buf.get_sequence()
    assert seq.shape == (10, 33, 3)


def test_dtw_runs_on_buffer_output(tmp_path: Path) -> None:
    make_reference_library(tmp_path, num_frames=10)
    engine = DTWEngine(tmp_path, window_len=10)

    buf = PoseBuffer(maxlen=10)
    for i in range(10):
        buf.add(make_reliable_frame(seed=i))

    window = [f.landmarks for f in list(buf._frames)]
    result = engine.compare(window)

    assert result is not None
    assert result.move_name == "windmill"
    assert result.distance >= 0
    engine.shutdown()


def test_async_dtw_resolves(tmp_path: Path) -> None:
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
    assert result.aligned is True
    engine.shutdown()


def test_full_pipeline_scores_when_keyframe_matches(tmp_path: Path) -> None:
    make_reference_library(tmp_path, num_frames=10)
    engine = DTWEngine(tmp_path, window_len=10)
    scorer = Scorer(tmp_path)

    buf = PoseBuffer(maxlen=10)
    for i in range(10):
        buf.add(make_reliable_frame(seed=i))

    window = [f.landmarks for f in list(buf._frames)]
    dtw_result = engine.compare(window)
    assert dtw_result is not None
    assert dtw_result.aligned is True

    # Ensure warping path maps live tail frame → keyframe frame 5
    live_idx = dtw_result.live_frames - 1
    dtw_result = DTWResult(
        aligned=dtw_result.aligned,
        distance=dtw_result.distance,
        normalized_distance=dtw_result.normalized_distance,
        move_name=dtw_result.move_name,
        threshold=dtw_result.threshold,
        live_frames=dtw_result.live_frames,
        reference_frames=dtw_result.reference_frames,
        live_path=tuple([live_idx] * 12),
        reference_path=tuple([5] * 12),
    )

    score_result = scorer.score(dtw_result, make_scorable_pose())
    assert score_result is not None
    assert score_result.move_name == "windmill"
    assert scorer.stats().total_attempts == 1
    engine.shutdown()


def test_combo_builds_on_perfect_streak(tmp_path: Path) -> None:
    make_reference_library(tmp_path, num_frames=10)
    engine = DTWEngine(tmp_path, window_len=10)
    scorer = Scorer(tmp_path)

    for attempt in range(3):
        buf = PoseBuffer(maxlen=10)
        for i in range(10):
            buf.add(make_reliable_frame(seed=i))
        window = [f.landmarks for f in list(buf._frames)]
        dtw_result = engine.compare(window)
        assert dtw_result is not None

        live_idx = dtw_result.live_frames - 1
        dtw_result = DTWResult(
            aligned=dtw_result.aligned,
            distance=dtw_result.distance,
            normalized_distance=dtw_result.normalized_distance,
            move_name=dtw_result.move_name,
            threshold=dtw_result.threshold,
            live_frames=dtw_result.live_frames,
            reference_frames=dtw_result.reference_frames,
            live_path=tuple([live_idx] * 12),
            reference_path=tuple([5] * 12),
        )
        scorer.score(dtw_result, make_scorable_pose())

    stats = scorer.stats()
    assert stats.total_attempts == 3
    assert stats.perfects == 3
    assert stats.streak == 3
    assert stats.total_points == pytest.approx(350.0)  # 100 + 150 + 100
    engine.shutdown()


def test_reset_clears_session(tmp_path: Path) -> None:
    scorer = Scorer(tmp_path)
    scorer.total_points = 500.0
    scorer.total_attempts = 10
    scorer.streak = 4
    scorer.reset()
    stats = scorer.stats()
    assert stats.total_points == 0.0
    assert stats.total_attempts == 0
    assert stats.streak == 0
