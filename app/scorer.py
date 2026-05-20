"""Scoring engine — keyframe thresholding, combo multiplier, and session stats."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from app.dtw_engine import DTWResult
from app.pose_estimator import PoseFrame

PERFECT = "Perfect"
CLOSE = "Close"
MISS = "Miss"

PERFECT_POINTS = 100.0
CLOSE_POINTS = 40.0
MISS_POINTS = 0.0

DEFAULT_THRESHOLD_DEG = 15.0
DEFAULT_KEYFRAME_OFFSET_TOLERANCE = 3


@dataclass(frozen=True)
class KeyframeResult:
    """Per-joint score outcome at a keyframe moment."""

    joint_name: str
    actual_angle: float
    target_angle: float
    diff: float
    tier: str
    points_earned: float


@dataclass(frozen=True)
class ScoreResult:
    """Score outcome for one scoring attempt."""

    results: list[KeyframeResult]
    points_this_attempt: float
    combo_multiplier_applied: float
    move_name: str


@dataclass(frozen=True)
class SessionStats:
    """Running totals for the current session."""

    total_points: float
    total_attempts: int
    perfects: int
    closes: int
    misses: int
    streak: int
    grade: str


@dataclass(frozen=True)
class JointTarget:
    """Reference target for one joint-angle measurement."""

    joint_name: str
    joint_triplet: tuple[int, int, int]
    target_angle: float
    threshold: float = DEFAULT_THRESHOLD_DEG


@dataclass(frozen=True)
class KeyframeTarget:
    """Reference keyframe with expected joint targets."""

    frame: int
    joints: list[JointTarget]


def joint_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Return angle in degrees at vertex ``b`` from three 2D points ``a, b, c``."""
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    c = np.asarray(c, dtype=np.float32).reshape(-1)
    if a.size < 2 or b.size < 2 or c.size < 2:
        raise ValueError("joint_angle requires points with at least 2 coordinates")

    a2 = a[:2]
    b2 = b[:2]
    c2 = c[:2]
    ba = a2 - b2
    bc = c2 - b2

    denom = float(np.linalg.norm(ba) * np.linalg.norm(bc))
    if denom <= 1e-8:
        raise ValueError("Cannot compute angle for near-zero segment length")

    cos_angle = float(np.dot(ba, bc) / denom)
    radians = float(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
    return float(np.degrees(radians))


def _score_tier(diff: float, threshold: float) -> tuple[str, float]:
    """Map absolute angular difference to a tier and raw points."""
    if diff <= threshold:
        return PERFECT, PERFECT_POINTS
    if diff <= 2.0 * threshold:
        return CLOSE, CLOSE_POINTS
    return MISS, MISS_POINTS


def _parse_triplet(value: Any) -> tuple[int, int, int]:
    """Parse a joint triplet from metadata."""
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"joint_triplet must have exactly 3 indices, got: {value!r}")
    triplet = tuple(int(i) for i in value)
    if any(i < 0 or i > 32 for i in triplet):
        raise ValueError(f"joint_triplet indices must be in [0, 32], got: {triplet}")
    return triplet  # type: ignore[return-value]


def _parse_joint_target(joint: dict[str, Any]) -> JointTarget:
    """Parse one joint target entry from keyframe metadata."""
    if not isinstance(joint, dict):
        raise ValueError(f"Joint target must be an object, got: {type(joint)}")

    name = str(joint.get("joint_name", "joint"))
    triplet_raw = joint.get("joint_triplet", joint.get("landmarks"))
    target_raw = joint.get("target_angle", joint.get("target_angle_deg"))
    threshold_raw = joint.get("threshold", joint.get("threshold_deg", DEFAULT_THRESHOLD_DEG))

    if triplet_raw is None or target_raw is None:
        raise ValueError(f"Joint target missing triplet or angle: {joint}")

    return JointTarget(
        joint_name=name,
        joint_triplet=_parse_triplet(triplet_raw),
        target_angle=float(target_raw),
        threshold=float(threshold_raw),
    )


def load_keyframe_targets(reference_dir: Path, move_name: str) -> list[KeyframeTarget]:
    """Load and validate keyframe targets from ``{move_name}_meta.json``."""
    meta_path = Path(reference_dir) / f"{move_name}_meta.json"
    if not meta_path.exists():
        return []

    with meta_path.open(encoding="utf-8") as f:
        meta = json.load(f)

    keyframes = meta.get("keyframes", [])
    if not isinstance(keyframes, list):
        raise ValueError(f"'keyframes' must be a list in {meta_path}")

    out: list[KeyframeTarget] = []
    for keyframe in keyframes:
        if not isinstance(keyframe, dict):
            raise ValueError(f"Each keyframe must be an object in {meta_path}")
        frame = int(keyframe.get("frame", -1))
        joints_raw = keyframe.get("joints", [])
        if frame < 0 or not isinstance(joints_raw, list):
            continue
        joints = [_parse_joint_target(j) for j in joints_raw]
        if joints:
            out.append(KeyframeTarget(frame=frame, joints=joints))
    return out


def _reference_index_for_live_frame(dtw_result: DTWResult, live_frame_index: int) -> Optional[int]:
    """Map a live frame index to a reference frame index via DTW warping path."""
    if not dtw_result.live_path or not dtw_result.reference_path:
        return None
    if len(dtw_result.live_path) != len(dtw_result.reference_path):
        return None

    aligned_refs = [
        ref_idx
        for live_idx, ref_idx in zip(dtw_result.live_path, dtw_result.reference_path)
        if live_idx == live_frame_index
    ]
    if not aligned_refs:
        return None
    return int(np.median(np.asarray(aligned_refs, dtype=np.float32)))


def _find_matching_keyframe(
    keyframes: list[KeyframeTarget],
    reference_index: Optional[int],
    *,
    max_offset: int = DEFAULT_KEYFRAME_OFFSET_TOLERANCE,
) -> Optional[KeyframeTarget]:
    """Pick the nearest keyframe if its frame index is within ``max_offset``."""
    if reference_index is None or not keyframes:
        return None
    nearest = min(keyframes, key=lambda k: abs(k.frame - reference_index))
    if abs(nearest.frame - reference_index) > max_offset:
        return None
    return nearest


class Scorer:
    """Score live poses at keyframe moments using DTW alignment and metadata targets."""

    def __init__(self, reference_dir: Path):
        self.reference_dir = Path(reference_dir)
        self.total_points: float = 0.0
        self.total_attempts: int = 0
        self.perfects: int = 0
        self.closes: int = 0
        self.misses: int = 0
        self.streak: int = 0

    def _combo_multiplier(self) -> float:
        """Return current combo multiplier from streak length."""
        if self.streak >= 5:
            return 2.0
        if self.streak >= 3:
            return 1.5
        return 1.0

    def _update_streak(self, attempt_tier: str) -> None:
        """Update streak according to the requested combo policy."""
        if attempt_tier == PERFECT:
            self.streak += 1
            return
        if attempt_tier == MISS:
            self.streak = 0

    def _attempt_tier(self, results: list[KeyframeResult]) -> str:
        """Collapse per-joint results into one attempt tier."""
        if not results:
            return MISS
        tiers = {r.tier for r in results}
        if MISS in tiers:
            return MISS
        if tiers == {PERFECT}:
            return PERFECT
        return CLOSE

    def score(self, dtw_result: DTWResult, pose_frame: PoseFrame) -> Optional[ScoreResult]:
        """Score one live frame against move keyframe targets; returns None when not scoreable."""
        if not dtw_result.aligned:
            return None

        keyframes = load_keyframe_targets(self.reference_dir, dtw_result.move_name)
        if not keyframes:
            return None

        live_index = max(dtw_result.live_frames - 1, 0)
        ref_index = _reference_index_for_live_frame(dtw_result, live_index)
        matched = _find_matching_keyframe(keyframes, ref_index)
        if matched is None:
            return None

        results: list[KeyframeResult] = []
        for target in matched.joints:
            i, j, k = target.joint_triplet
            landmarks = pose_frame.landmarks
            if landmarks.ndim != 2 or landmarks.shape[0] < 33 or landmarks.shape[1] < 2:
                raise ValueError(f"Expected landmarks shape (33, >=2), got {landmarks.shape}")

            angle = joint_angle(landmarks[i], landmarks[j], landmarks[k])
            diff = abs(angle - target.target_angle)
            tier, points = _score_tier(diff, target.threshold)
            results.append(
                KeyframeResult(
                    joint_name=target.joint_name,
                    actual_angle=angle,
                    target_angle=target.target_angle,
                    diff=diff,
                    tier=tier,
                    points_earned=points,
                )
            )

        attempt_tier = self._attempt_tier(results)
        self._update_streak(attempt_tier)
        combo = self._combo_multiplier()
        raw_points = float(sum(r.points_earned for r in results))
        scored_points = raw_points * combo

        self.total_attempts += 1
        self.total_points += scored_points
        if attempt_tier == PERFECT:
            self.perfects += 1
        elif attempt_tier == CLOSE:
            self.closes += 1
        else:
            self.misses += 1

        return ScoreResult(
            results=results,
            points_this_attempt=scored_points,
            combo_multiplier_applied=combo,
            move_name=dtw_result.move_name,
        )

    def grade(self) -> str:
        """Return session letter grade by perfect-rate thresholds."""
        if self.total_attempts <= 0:
            return "D"
        perfect_rate = self.perfects / self.total_attempts
        if perfect_rate >= 0.90:
            return "S"
        if perfect_rate >= 0.75:
            return "A"
        if perfect_rate >= 0.60:
            return "B"
        if perfect_rate >= 0.40:
            return "C"
        return "D"

    def stats(self) -> SessionStats:
        """Return immutable snapshot of current session totals and grade."""
        return SessionStats(
            total_points=self.total_points,
            total_attempts=self.total_attempts,
            perfects=self.perfects,
            closes=self.closes,
            misses=self.misses,
            streak=self.streak,
            grade=self.grade(),
        )

    def reset(self) -> None:
        """Clear all session totals and combo streak."""
        self.total_points = 0.0
        self.total_attempts = 0
        self.perfects = 0
        self.closes = 0
        self.misses = 0
        self.streak = 0

if __name__ == "__main__":
    import cv2
    from pathlib import Path
    from app.pose_estimator import PoseEstimator
    from app.buffer import PoseBuffer
    from app.dtw_engine import DTWEngine
    from app.pose_estimator import PoseEstimator, is_pose_reliable
    
    REFERENCE_DIR = Path("reference_moves")

    print("Loading models...")
    dummy = np.zeros((480, 640, 3), dtype=np.uint8)

    with PoseEstimator(model_complexity=0) as estimator:
        estimator.process(dummy)  # warmup
        print("Ready — press Q to quit")

        buf    = PoseBuffer()
        engine = DTWEngine(REFERENCE_DIR)
        scorer = Scorer(REFERENCE_DIR)
        future = None

        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pose = estimator.process(rgb)

            # ── pipeline ──────────────────────────────────────────
            if pose is not None and is_pose_reliable(pose):
                buf.add(pose)

            # fire async DTW every 3 frames once buffer is ready
            if buf.is_ready() and future is None:
                window = [f.landmarks for f in list(buf._frames)]
                future = engine.compare_async(window)

            # collect result when ready
            if future is not None and future.done():
                dtw_result = future.result()
                future = None   # reset for next comparison

                if dtw_result and dtw_result.aligned:
                    score_result = scorer.score(dtw_result, pose)

            # ── overlay ───────────────────────────────────────────
            stats = scorer.stats()
            cv2.putText(frame, f"Score: {stats.total_points}",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (0, 255, 0), 2)
            cv2.putText(frame, f"Grade: {stats.grade}",
                        (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (0, 255, 255), 2)
            cv2.putText(frame, f"Streak: {stats.streak}",
                        (20, 120), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (255, 165, 0), 2)
            cv2.putText(frame, f"Buffer: {len(buf)}/60",
                        (20, 160), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (200, 200, 200), 1)

            if engine.has_reference:
                cv2.putText(frame, f"Move: {engine.move_name}",
                            (20, 200), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (200, 200, 200), 1)
            else:
                cv2.putText(frame, "No reference moves loaded",
                            (20, 200), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 0, 255), 1)

            cv2.imshow("Breakdance Coach — Pipeline Test", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()
        engine.shutdown()