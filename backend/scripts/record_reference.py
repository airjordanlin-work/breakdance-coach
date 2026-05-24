"""
Record a reference move from your webcam and save it to reference_moves/.

Usage:
    python3 scripts/record_reference.py

Instructions:
    1. Stand in frame so your full body is visible
    2. Press SPACE to start recording
    3. Hold your T-Pose for 2 seconds
    4. Recording stops automatically after 60 frames
    5. Files saved to reference_moves/t_pose.npy + t_pose_meta.json
"""

import json
import cv2
import numpy as np
from pathlib import Path
from app.pose_estimator import PoseEstimator, is_pose_reliable

MOVE_NAME    = "t_pose"
OUTPUT_DIR   = Path("reference_moves")
WINDOW_LEN   = 60
WARMUP_DUMMY = np.zeros((480, 640, 3), dtype=np.uint8)

# MediaPipe landmark indices for the joints we care about
# Left arm:  shoulder(11) → elbow(13) → wrist(15)
# Right arm: shoulder(12) → elbow(14) → wrist(16)
KEYFRAME_META = {
    "name": MOVE_NAME,
    "dtw_threshold": 300.0,
    "keyframes": [
        {
            "frame": 30,   # peak of the hold — middle of the recording
            "joints": [
                {
                    "joint_name": "left_elbow",
                    "joint_triplet": [11, 13, 15],   # shoulder→elbow→wrist
                    "target_angle": 170.0,            # nearly straight arm
                    "threshold": 15.0
                },
                {
                    "joint_name": "right_elbow",
                    "joint_triplet": [12, 14, 16],
                    "target_angle": 170.0,
                    "threshold": 15.0
                }
            ]
        }
    ]
}


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    frames_collected = []
    recording = False

    print("=== Reference Move Recorder ===")
    print(f"Move: {MOVE_NAME}")
    print("Stand in frame so your FULL BODY is visible")
    print("Press SPACE to start recording your T-Pose")
    print("Hold both arms straight out to your sides")
    print("Press Q to quit\n")

    with PoseEstimator(model_complexity=0) as estimator:
        estimator.process(WARMUP_DUMMY)

        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pose = estimator.process(rgb, keep_raw=True)

            reliable = pose is not None and is_pose_reliable(pose)

            # ── overlay ───────────────────────────────────────────
            status_color = (0, 255, 0) if reliable else (0, 0, 255)
            status_text  = "Body detected" if reliable else "Stand in frame"
            cv2.putText(frame, status_text, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 2)

            if recording:
                count = len(frames_collected)
                cv2.putText(frame, f"RECORDING: {count}/{WINDOW_LEN}",
                            (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (0, 0, 255), 2)
                cv2.putText(frame, "Hold your T-Pose!",
                            (20, 120), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 165, 255), 2)

                if reliable:
                    frames_collected.append(pose.landmarks.copy())

                if len(frames_collected) >= WINDOW_LEN:
                    print(f"\nRecording complete — {WINDOW_LEN} frames captured")
                    break
            else:
                cv2.putText(frame, "Press SPACE to record T-Pose",
                            (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (200, 200, 200), 2)
                cv2.putText(frame, "Extend BOTH arms straight out",
                            (20, 120), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (200, 200, 200), 1)

            cv2.imshow("Reference Recorder", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(" ") and reliable and not recording:
                print("Recording started — hold your T-Pose!")
                recording = True
                frames_collected = []
            elif key == ord("q"):
                print("Quit — no files saved")
                cap.release()
                cv2.destroyAllWindows()
                return

        cap.release()
        cv2.destroyAllWindows()

    if len(frames_collected) < WINDOW_LEN:
        print("Not enough frames recorded — try again")
        return

    # ── save files ────────────────────────────────────────────────
    sequence = np.stack(frames_collected, axis=0)          # (60, 33, 3)
    flat_seq = sequence.reshape(WINDOW_LEN, -1)            # (60, 99)

    npy_path  = OUTPUT_DIR / f"{MOVE_NAME}.npy"
    meta_path = OUTPUT_DIR / f"{MOVE_NAME}_meta.json"

    np.save(npy_path, flat_seq)
    with meta_path.open("w") as f:
        json.dump(KEYFRAME_META, f, indent=2)

    print(f"\n✓ Saved {npy_path}  shape={flat_seq.shape}")
    print(f"✓ Saved {meta_path}")
    print(f"\nNow run:  python3 -m app.scorer")
    print("Hold a T-Pose in frame and watch the score update!")


if __name__ == "__main__":
    main()