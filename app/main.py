"""Entry point — webcam or video loop with pose, DTW, and scoring."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow `python app/main.py` from repo root
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
import asyncio
from collections import deque
import cv2
import numpy as np

from app.dtw_engine import DEFAULT_WINDOW_LEN, DTWEngine, DTWResult
from app.overlay import draw_hud, draw_skeleton
from app.pose_estimator import PoseEstimator
from app.scorer import Scorer, default_windmill_targets

WINDOW_LEN = DEFAULT_WINDOW_LEN
DTW_EVERY_N_FRAMES = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Breakdance Coach — real-time move feedback")
    parser.add_argument(
        "--source",
        default="webcam",
        help="webcam, path to video (e.g. demo/sample.mp4), or 0 for default camera",
    )
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=Path("reference_moves"),
        help="Directory containing .npy reference sequences",
    )
    parser.add_argument("--record", action="store_true", help="Record keypoints to --output")
    parser.add_argument("--output", type=Path, help="Base path for recorded .npy (no extension)")
    return parser.parse_args()


def open_capture(source: str) -> cv2.VideoCapture:
    if source == "webcam":
        return cv2.VideoCapture(0)
    if source.isdigit():
        return cv2.VideoCapture(int(source))
    return cv2.VideoCapture(source)


async def run_loop(args: argparse.Namespace) -> None:
    cap = open_capture(args.source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video source: {args.source}")

    estimator = PoseEstimator()
    dtw = DTWEngine(args.reference_dir)
    scorer = Scorer()
    targets = default_windmill_targets()

    buffer: deque[np.ndarray] = deque(maxlen=WINDOW_LEN)
    recorded: list[np.ndarray] = []
    frame_idx = 0
    last_dtw: DTWResult | None = None
    last_score_result = None
    pending_dtw = None

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pose = estimator.process(rgb)

            if pose is not None:
                buffer.append(pose.landmarks)
                if args.record:
                    recorded.append(pose.landmarks.copy())
                draw_skeleton(frame, pose.landmarks, pose.visibility)

                # Scoring runs only when DTW says the sequence is aligned (flowchart H → I)
                if last_dtw is not None and last_dtw.aligned:
                    last_score_result = scorer.evaluate(pose.landmarks, targets)

            if (
                frame_idx % DTW_EVERY_N_FRAMES == 0
                and pending_dtw is None
            ):
                pending_dtw = dtw.compare_async(list(buffer))

            if pending_dtw is not None and pending_dtw.done():
                try:
                    last_dtw = pending_dtw.result()
                except Exception:
                    last_dtw = None
                pending_dtw = None

            dtw_distance = last_dtw.distance if last_dtw else None
            draw_hud(
                frame,
                score=scorer.total_score,
                grade=scorer.grade(),
                move_name=dtw.move_name,
                dtw_distance=dtw_distance,
                last_result=last_score_result,
                dtw_aligned=last_dtw.aligned if last_dtw else None,
            )

            cv2.imshow("Breakdance Coach", frame)
            frame_idx += 1
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            await asyncio.sleep(0)
    finally:
        cap.release()
        estimator.close()
        dtw.shutdown()
        cv2.destroyAllWindows()

        if args.record and args.output and recorded:
            out = Path(args.output)
            out.parent.mkdir(parents=True, exist_ok=True)
            np.save(f"{out}.npy", np.stack(recorded))
            print(f"Saved {len(recorded)} frames to {out}.npy")


def main() -> None:
    args = parse_args()
    asyncio.run(run_loop(args))


if __name__ == "__main__":
    main()
