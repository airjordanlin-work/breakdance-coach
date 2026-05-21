"""Entry point — webcam/video loop wiring pose, buffer, DTW, scoring, and overlay."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

# Allow `python app/main.py` from repo root
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app import overlay
from app.buffer import PoseBuffer, WINDOW_LEN
from app.dtw_engine import DTWEngine
from app.pose_estimator import PoseEstimator, is_pose_reliable
from app.scorer import Scorer

DTW_EVERY_N_FRAMES = 3
KEYFRAME_FLASH_FRAMES = 30


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for source, reference library, and display options."""
    parser = argparse.ArgumentParser(description="Breakdance Coach — real-time move feedback")
    parser.add_argument(
        "--source",
        default="webcam",
        help='Webcam (default) or path to a video file (e.g. demo/sample.mp4)',
    )
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=Path("reference_moves"),
        help="Directory containing reference .npy sequences and metadata JSON",
    )
    parser.add_argument(
        "--model-complexity",
        type=int,
        choices=[0, 1, 2],
        default=0,
        help="MediaPipe Pose model complexity (0=fast, 2=accurate)",
    )
    parser.add_argument(
        "--window-name",
        default="Breakdance Coach",
        help="OpenCV display window title",
    )
    return parser.parse_args()


def open_capture(source: str) -> cv2.VideoCapture:
    """Open webcam device 0 or a video file path."""
    if source == "webcam":
        return cv2.VideoCapture(0)
    if source.isdigit():
        return cv2.VideoCapture(int(source))
    return cv2.VideoCapture(source)


def warn_if_no_references(reference_dir: Path) -> None:
    """Print a warning when the reference move library is empty."""
    if not any(reference_dir.glob("*.npy")):
        print(f"Warning: no reference moves found in {reference_dir} — DTW will not align.")


def print_session_summary(scorer: Scorer) -> None:
    """Print final session totals to the terminal."""
    stats = scorer.stats()
    print("Session complete")
    print(f"Final score: {stats.total_points:.0f}")
    print(f"Grade: {stats.grade}")
    print(f"Perfect: {stats.perfects} / Close: {stats.closes} / Miss: {stats.misses}")


def run_loop(args: argparse.Namespace) -> None:
    """Main render loop: capture → pose → buffer → DTW → score → overlay → display."""
    reference_dir = Path(args.reference_dir)
    warn_if_no_references(reference_dir)

    print("Loading pose model...")
    dummy = np.zeros((480, 640, 3), dtype=np.uint8)

    cap = open_capture(args.source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video source: {args.source}")

    estimator = PoseEstimator(model_complexity=args.model_complexity)
    estimator.process(dummy)  # warmup before live frames
    print("Ready — press Q to quit")

    engine = DTWEngine(reference_dir)
    scorer = Scorer(reference_dir)
    buf = PoseBuffer()

    future = None
    flash_counter = 0
    score_result = None
    dtw_result = None
    frame_idx = 0
    pose_frame = None

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pose = estimator.process(rgb, keep_raw=True)

            if pose is not None and is_pose_reliable(pose):
                pose_frame = pose
                buf.add(pose)

            if buf.is_ready() and frame_idx % DTW_EVERY_N_FRAMES == 0 and future is None:
                window = [f.landmarks for f in buf._frames]
                future = engine.compare_async(window)

            if future is not None and future.done():
                try:
                    dtw_result = future.result()
                except Exception:
                    dtw_result = None
                future = None

                if dtw_result is not None and dtw_result.aligned and pose_frame is not None:
                    new_score = scorer.score(dtw_result, pose_frame)
                    if new_score is not None:
                        score_result = new_score
                        flash_counter = KEYFRAME_FLASH_FRAMES

            fill_ratio = len(buf) / WINDOW_LEN
            move_name = dtw_result.move_name if dtw_result else engine.move_name

            frame, flash_counter = overlay.render(
                frame,
                pose_frame=pose_frame,
                dtw_result=dtw_result,
                score_result=score_result,
                stats=scorer.stats(),
                fill_ratio=fill_ratio,
                flash_counter=flash_counter,
                move_name=move_name,
            )

            cv2.imshow(args.window_name, frame)
            frame_idx += 1

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        estimator.close()
        engine.shutdown()
        print_session_summary(scorer)


def main() -> None:
    """Program entry: parse args and run the live coaching loop."""
    args = parse_args()
    run_loop(args)


if __name__ == "__main__":
    main()
