"""
Preview AIST++ reference moves as animated skeleton on screen.
Use this to identify what each move looks like and rename them.

Usage:
    python3 scripts/preview_moves.py
    python3 scripts/preview_moves.py --dir reference_moves/
    
Controls:
    SPACE — next move
    B     — previous move
    R     — rename current move
    Q     — quit
"""

from __future__ import annotations

import sys
import json
import time
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# MediaPipe body connections
CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (24, 26), (26, 28),
]

MEDIAPIPE_NAMES = {
    0: "nose", 11: "L.shoulder", 12: "R.shoulder",
    13: "L.elbow", 14: "R.elbow", 15: "L.wrist", 16: "R.wrist",
    23: "L.hip", 24: "R.hip", 25: "L.knee", 26: "R.knee",
    27: "L.ankle", 28: "R.ankle",
}

W, H = 800, 600
SCALE = 120
CX, CY = W // 2, int(H * 0.55)


def draw_frame(canvas: np.ndarray, flat_frame: np.ndarray, frame_idx: int,
               total: int, move_name: str, source_file: str) -> None:
    canvas[:] = (15, 15, 20)
    landmarks = flat_frame.reshape(33, 3)

    # project 3D → 2D using x and y only
    points: dict[int, tuple[int, int]] = {}
    for i in range(33):
        x = int(CX + landmarks[i, 0] * SCALE)
        y = int(CY + landmarks[i, 1] * SCALE)
        points[i] = (x, y)

    # draw connections
    for a, b in CONNECTIONS:
        if a in points and b in points:
            cv2.line(canvas, points[a], points[b], (60, 180, 180), 2, cv2.LINE_AA)

    # draw joints
    for i, (px, py) in points.items():
        if i in MEDIAPIPE_NAMES:
            cv2.circle(canvas, (px, py), 6, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(canvas, (px, py), 4, (0, 220, 180), -1, cv2.LINE_AA)

    # labels
    cv2.putText(canvas, move_name.upper(), (20, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (240, 240, 240), 2, cv2.LINE_AA)
    cv2.putText(canvas, source_file, (20, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 120, 120), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"frame {frame_idx + 1} / {total}", (20, H - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100, 100, 100), 1, cv2.LINE_AA)
    cv2.putText(canvas, "SPACE=next  B=prev  R=rename  Q=quit",
                (W - 340, H - 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.38, (80, 80, 80), 1, cv2.LINE_AA)


def rename_move(npy_path: Path, meta_path: Path, new_name: str) -> None:
    """Rename the move files and update the meta JSON."""
    parent = npy_path.parent
    new_npy  = parent / f"{new_name}.npy"
    new_meta = parent / f"{new_name}_meta.json"

    npy_path.rename(new_npy)

    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        meta["name"] = new_name
        new_meta.write_text(json.dumps(meta, indent=2))
        meta_path.unlink(missing_ok=True)

    print(f"Renamed to: {new_name}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=Path, default=Path("reference_moves"))
    args = parser.parse_args()

    npy_files = sorted(args.dir.glob("*.npy"))
    if not npy_files:
        raise SystemExit(f"No .npy files found in {args.dir}")

    print(f"Found {len(npy_files)} reference moves")
    print("Controls: SPACE=next  B=prev  R=rename  Q=quit\n")

    canvas   = np.zeros((H, W, 3), dtype=np.uint8)
    move_idx = 0
    frame_idx = 0
    playing  = True

    cv2.namedWindow("Reference Move Preview", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Reference Move Preview", W, H)

    while True:
        npy_path  = npy_files[move_idx]
        meta_path = npy_path.parent / npy_path.name.replace(".npy", "_meta.json")
        sequence  = np.load(npy_path)   # (60, 99)
        move_name = npy_path.stem

        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            source = meta.get("original_file", "")
        else:
            source = ""

        draw_frame(canvas, sequence[frame_idx], frame_idx,
                   len(sequence), move_name, source)
        cv2.imshow("Reference Move Preview", canvas)

        key = cv2.waitKey(80) & 0xFF   # ~12fps playback

        if key == ord("q"):
            break
        elif key == ord(" "):
            move_idx = (move_idx + 1) % len(npy_files)
            frame_idx = 0
            print(f"Move {move_idx+1}/{len(npy_files)}: {npy_files[move_idx].stem}")
        elif key == ord("b"):
            move_idx = (move_idx - 1) % len(npy_files)
            frame_idx = 0
            print(f"Move {move_idx+1}/{len(npy_files)}: {npy_files[move_idx].stem}")
        elif key == ord("r"):
            cv2.destroyAllWindows()
            new_name = input(f"Rename '{move_name}' to: ").strip()
            if new_name:
                rename_move(npy_path, meta_path, new_name)
                npy_files = sorted(args.dir.glob("*.npy"))
                move_idx = min(move_idx, len(npy_files) - 1)
            cv2.namedWindow("Reference Move Preview", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Reference Move Preview", W, H)
        else:
            # auto advance frames
            if playing:
                frame_idx = (frame_idx + 1) % len(sequence)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()