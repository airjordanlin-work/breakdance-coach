"""Keyframe angle scoring and combo multiplier."""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

# MediaPipe Pose landmark indices used for angle checks
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26


def joint_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle at vertex b formed by segments ba and bc, in degrees."""
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


@dataclass
class KeyframeTarget:
    name: str
    joint_triplet: tuple[int, int, int]
    target_angle_deg: float
    tolerance_deg: float = 15.0


@dataclass
class ScoreResult:
    hit: bool
    points: int
    feedback: list[str] = field(default_factory=list)
    combo_multiplier: int = 1


@dataclass
class Scorer:
    """Threshold-based keyframe scorer with rolling combo."""

    combo_streak: int = 0
    total_score: int = 0
    window_size: int = 10
    _recent_hits: list[bool] = field(default_factory=list)

    def evaluate(self, landmarks: np.ndarray, targets: list[KeyframeTarget]) -> ScoreResult:
        feedback: list[str] = []
        hits = 0

        for target in targets:
            i, j, k = target.joint_triplet
            angle = joint_angle(landmarks[i], landmarks[j], landmarks[k])
            delta = abs(angle - target.target_angle_deg)
            if delta <= target.tolerance_deg:
                hits += 1
            else:
                direction = "extend" if angle < target.target_angle_deg else "bend"
                feedback.append(f"{target.name}: {direction} (~{delta:.0f}° off)")

        hit = hits == len(targets) and len(targets) > 0
        self._recent_hits.append(hit)
        if len(self._recent_hits) > self.window_size:
            self._recent_hits.pop(0)

        if hit:
            self.combo_streak += 1
            multiplier = min(self.combo_streak, 5)
            points = 100 * multiplier
            self.total_score += points
        else:
            self.combo_streak = 0
            multiplier = 1
            points = 0

        return ScoreResult(
            hit=hit,
            points=points,
            feedback=feedback,
            combo_multiplier=multiplier if hit else 1,
        )

    def grade(self) -> str:
        """Letter grade from recent hit rate."""
        if not self._recent_hits:
            return "—"
        rate = sum(self._recent_hits) / len(self._recent_hits)
        if rate >= 0.8:
            return "A"
        if rate >= 0.6:
            return "B"
        if rate >= 0.4:
            return "C"
        if rate >= 0.2:
            return "D"
        return "F"


def default_windmill_targets() -> list[KeyframeTarget]:
    """Starter targets when no meta JSON is present."""
    return [
        KeyframeTarget("left_elbow", (L_SHOULDER, L_ELBOW, L_WRIST), target_angle_deg=90.0),
        KeyframeTarget("right_elbow", (R_SHOULDER, R_ELBOW, R_WRIST), target_angle_deg=90.0),
    ]
