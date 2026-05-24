"""
Convert AIST++ 3D keypoints to .npy reference moves for Breakdance Coach.

Usage:
    python3 scripts/convert_aist.py \
        --keypoints-dir /Users/jordanlin/keypoints3d \
        --output reference_moves/ \
        --genre BR \
        --target-count 10 \
        --skip 10
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

COCO_TO_MEDIAPIPE: dict[int, int] = {
    0:  0,    # nose
    1:  2,    # left_eye
    2:  5,    # right_eye
    3:  7,    # left_ear
    4:  8,    # right_ear
    5:  11,   # left_shoulder
    6:  12,   # right_shoulder
    7:  13,   # left_elbow
    8:  14,   # right_elbow
    9:  15,   # left_wrist
    10: 16,   # right_wrist
    11: 23,   # left_hip
    12: 24,   # right_hip
    13: 25,   # left_knee
    14: 26,   # right_knee
    15: 27,   # left_ankle
    16: 28,   # right_ankle
}

TARGET_FRAMES = 60
L_HIP_MP, R_HIP_MP = 23, 24
MAX_VALID_VALUE = 35.0   # anything beyond this is corrupted data


def load_keypoints(pkl_path: Path) -> np.ndarray:
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    key = "keypoints3d_optim" if "keypoints3d_optim" in data else "keypoints3d"
    return data[key].astype(np.float32)


def coco_to_mediapipe(coco_joints: np.ndarray) -> np.ndarray:
    T = coco_joints.shape[0]
    mp = np.zeros((T, 33, 3), dtype=np.float32)
    for coco_idx, mp_idx in COCO_TO_MEDIAPIPE.items():
        mp[:, mp_idx, :] = coco_joints[:, coco_idx, :]
    return mp


def normalize(landmarks: np.ndarray) -> np.ndarray:
    out = landmarks.copy()
    for t in range(len(out)):
        hip_center = (out[t, L_HIP_MP] + out[t, R_HIP_MP]) / 2.0
        out[t] -= hip_center
        hip_dist = float(np.linalg.norm(
            out[t, L_HIP_MP, :2] - out[t, R_HIP_MP, :2]
        ))
        scale = max(hip_dist, 1e-6)
        out[t] /= scale
    return out


def resample(seq: np.ndarray, target: int = TARGET_FRAMES) -> np.ndarray:
    T = seq.shape[0]
    if T == target:
        return seq
    indices   = np.linspace(0, T - 1, target)
    resampled = np.zeros((target, *seq.shape[1:]), dtype=np.float32)
    for i, idx in enumerate(indices):
        lo = int(idx); hi = min(lo + 1, T - 1); t = idx - lo
        resampled[i] = seq[lo] * (1 - t) + seq[hi] * t
    return resampled


def convert_file(
    pkl_path: Path,
    output_dir: Path,
    move_name: str,
    dtw_threshold: float = 2000.0,
) -> bool:
    try:
        raw        = load_keypoints(pkl_path)
        mp_joints  = coco_to_mediapipe(raw)
        normalized = normalize(mp_joints)
        resampled  = resample(normalized, TARGET_FRAMES)
        flat       = resampled.reshape(TARGET_FRAMES, -1)

        # reject corrupted sequences
        max_val = float(abs(flat).max())
        if max_val > MAX_VALID_VALUE:
            print(f"  SKIPPED — corrupted data (max={max_val:.1f})")
            return False

        npy_path  = output_dir / f"{move_name}.npy"
        meta_path = output_dir / f"{move_name}_meta.json"

        np.save(npy_path, flat)

        meta = {
            "name":          move_name,
            "dtw_threshold": dtw_threshold,
            "source":        "AIST++",
            "original_file": pkl_path.name,
            "keyframes": [
                {
                    "frame": 59,
                    "joints": [
                        {
                            "joint_name":    "left_elbow",
                            "joint_triplet": [11, 13, 15],
                            "target_angle":  160.0,
                            "threshold":     30.0
                        },
                        {
                            "joint_name":    "right_elbow",
                            "joint_triplet": [12, 14, 16],
                            "target_angle":  160.0,
                            "threshold":     30.0
                        }
                    ]
                }
            ]
        }

        with meta_path.open("w") as f:
            json.dump(meta, f, indent=2)

        print(f"  OK  {npy_path.name}  max={max_val:.2f}  ({raw.shape[0]} frames → {TARGET_FRAMES})")
        return True

    except Exception as e:
        print(f"  FAILED {pkl_path.name}: {e}")
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--keypoints-dir", type=Path,
                        default=Path("/Users/jordanlin/keypoints3d"))
    parser.add_argument("--output",        type=Path,
                        default=Path("reference_moves"))
    parser.add_argument("--genre",         default="BR")
    parser.add_argument("--target-count",  type=int, default=10,
                        help="How many clean sequences to collect")
    parser.add_argument("--skip",          type=int, default=0,
                        help="Skip first N sequences (already processed)")
    parser.add_argument("--threshold",     type=float, default=2000.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    all_files = sorted(args.keypoints_dir.glob(f"g{args.genre}_*.pkl"))
    if not all_files:
        raise SystemExit(f"No files found for genre '{args.genre}'")

    # count existing moves to continue numbering correctly
    existing = len(list(args.output.glob(f"{args.genre.lower()}_*.npy")))
    next_num = existing + 1

    files_to_try = all_files[args.skip:]
    print(f"Found {len(all_files)} total {args.genre} sequences")
    print(f"Skipping first {args.skip}, trying up to {len(files_to_try)} more")
    print(f"Target: {args.target_count} clean sequences (have {existing} already)\n")

    converted = 0
    tried     = 0

    for pkl_path in files_to_try:
        if converted >= args.target_count:
            break
        tried += 1
        move_name = f"{args.genre.lower()}_{next_num:02d}"
        print(f"[trying {tried}] {pkl_path.name}  →  {move_name}")
        if convert_file(pkl_path, args.output, move_name, args.threshold):
            converted += 1
            next_num  += 1

    total = existing + converted
    print(f"\nDone — added {converted} clean sequences ({total} total in {args.output}/)")


if __name__ == "__main__":
    main()