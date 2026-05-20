"""Tests for keyframe scoring and combo logic."""

import numpy as np
import pytest

from app.scorer import KeyframeTarget, Scorer, joint_angle


def test_joint_angle_right_angle():
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 0.0, 0.0])
    c = np.array([0.0, 1.0, 0.0])
    assert joint_angle(a, b, c) == pytest.approx(90.0, abs=0.1)


def test_scorer_hit_awards_points_and_combo():
    scorer = Scorer()
    landmarks = np.zeros((33, 3), dtype=np.float32)

    # 90° at left elbow: shoulder (1,0), elbow origin, wrist (0,1)
    landmarks[11] = [1.0, 0.0, 0.0]
    landmarks[13] = [0.0, 0.0, 0.0]
    landmarks[15] = [0.0, 1.0, 0.0]
    landmarks[12] = [-1.0, 0.0, 0.0]
    landmarks[14] = [0.0, 0.0, 0.0]
    landmarks[16] = [0.0, 1.0, 0.0]

    targets = [
        KeyframeTarget("left_elbow", (11, 13, 15), target_angle_deg=90.0, tolerance_deg=5.0),
        KeyframeTarget("right_elbow", (12, 14, 16), target_angle_deg=90.0, tolerance_deg=5.0),
    ]

    r1 = scorer.evaluate(landmarks, targets)
    assert r1.hit is True
    assert r1.points == 100

    r2 = scorer.evaluate(landmarks, targets)
    assert r2.hit is True
    assert r2.combo_multiplier == 2
    assert r2.points == 200


def test_scorer_miss_resets_combo():
    scorer = Scorer()
    landmarks = np.zeros((33, 3), dtype=np.float32)
    scorer.combo_streak = 3

    # Collinear points → ~180°, well outside 90° ± 1°
    landmarks[11] = [1.0, 0.0, 0.0]
    landmarks[13] = [0.0, 0.0, 0.0]
    landmarks[15] = [-1.0, 0.0, 0.0]

    targets = [
        KeyframeTarget("left_elbow", (11, 13, 15), target_angle_deg=90.0, tolerance_deg=1.0),
    ]
    result = scorer.evaluate(landmarks, targets)
    assert result.hit is False
    assert scorer.combo_streak == 0
