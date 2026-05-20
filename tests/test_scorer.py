"""Tests for keyframe scoring, combo logic, and session stats."""

import numpy as np
import pytest

from app.scorer import (
    CLOSE,
    CLOSE_POINTS,
    MISS,
    MISS_POINTS,
    PERFECT,
    PERFECT_POINTS,
    Scorer,
    SessionStats,
    _score_tier,
    joint_angle,
)


def test_joint_angle_right_angle() -> None:
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 0.0, 0.0])
    c = np.array([0.0, 1.0, 0.0])
    assert joint_angle(a, b, c) == pytest.approx(90.0, abs=0.1)


def test_joint_angle_straight_line() -> None:
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 0.0, 0.0])
    c = np.array([-1.0, 0.0, 0.0])
    assert joint_angle(a, b, c) == pytest.approx(180.0, abs=0.1)


def test_joint_angle_raises_for_zero_segment() -> None:
    a = np.array([0.0, 0.0, 0.0])
    b = np.array([0.0, 0.0, 0.0])
    c = np.array([1.0, 0.0, 0.0])
    with pytest.raises(ValueError):
        joint_angle(a, b, c)


def test_score_tier_perfect() -> None:
    tier, points = _score_tier(diff=5.0, threshold=10.0)
    assert tier == PERFECT
    assert points == PERFECT_POINTS


def test_score_tier_close() -> None:
    tier, points = _score_tier(diff=15.0, threshold=10.0)
    assert tier == CLOSE
    assert points == CLOSE_POINTS


def test_score_tier_miss() -> None:
    tier, points = _score_tier(diff=25.0, threshold=10.0)
    assert tier == MISS
    assert points == MISS_POINTS


def test_combo_multiplier_thresholds(tmp_path) -> None:
    scorer = Scorer(tmp_path)
    scorer.streak = 0
    assert scorer._combo_multiplier() == 1.0
    scorer.streak = 3
    assert scorer._combo_multiplier() == 1.5
    scorer.streak = 5
    assert scorer._combo_multiplier() == 2.0


def test_update_streak_rules(tmp_path) -> None:
    scorer = Scorer(tmp_path)
    scorer._update_streak(PERFECT)
    assert scorer.streak == 1
    scorer._update_streak(CLOSE)
    assert scorer.streak == 1
    scorer._update_streak(MISS)
    assert scorer.streak == 0


def test_grade_thresholds(tmp_path) -> None:
    scorer = Scorer(tmp_path)
    assert scorer.grade() == "D"

    scorer.total_attempts = 10
    scorer.perfects = 9
    assert scorer.grade() == "S"

    scorer.perfects = 8
    assert scorer.grade() == "A"

    scorer.perfects = 6
    assert scorer.grade() == "B"

    scorer.perfects = 4
    assert scorer.grade() == "C"

    scorer.perfects = 3
    assert scorer.grade() == "D"


def test_stats_snapshot(tmp_path) -> None:
    scorer = Scorer(tmp_path)
    stats = scorer.stats()
    assert isinstance(stats, SessionStats)
    assert stats.total_points == 0.0
    assert stats.total_attempts == 0
    assert stats.grade == "D"


def test_reset_clears_state(tmp_path) -> None:
    scorer = Scorer(tmp_path)
    scorer.total_points = 500.0
    scorer.total_attempts = 5
    scorer.perfects = 2
    scorer.closes = 1
    scorer.misses = 2
    scorer.streak = 3
    scorer.reset()

    stats = scorer.stats()
    assert stats.total_points == 0.0
    assert stats.total_attempts == 0
    assert stats.perfects == 0
    assert stats.closes == 0
    assert stats.misses == 0
    assert stats.streak == 0