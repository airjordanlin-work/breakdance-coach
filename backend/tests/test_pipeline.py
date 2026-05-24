"""Quick pipeline test without webcam — run from project root."""

import numpy as np
from pathlib import Path
from app.dtw_engine import DTWEngine
from app.scorer import Scorer
from app.pose_estimator import PoseFrame

# fake T-pose landmarks in normalized body space
landmarks = np.zeros((33, 3), dtype=np.float32)
landmarks[11] = [-0.5, -1.45, 0]   # left shoulder
landmarks[12] = [ 0.5, -1.45, 0]   # right shoulder
landmarks[23] = [-0.2, -0.80, 0]   # left hip
landmarks[24] = [ 0.2, -0.80, 0]   # right hip
landmarks[13] = [-1.1, -1.45, 0]   # left elbow — T-pose arms out
landmarks[14] = [ 1.1, -1.45, 0]   # right elbow
landmarks[15] = [-1.65,-1.45, 0]   # left wrist
landmarks[16] = [ 1.65,-1.45, 0]   # right wrist
visibility = np.ones(33, dtype=np.float32)

fake_frame = PoseFrame(landmarks=landmarks, visibility=visibility)
engine     = DTWEngine(Path("reference_moves"))
scorer     = Scorer(Path("reference_moves"))

print(f"Library loaded: {len(engine.library)} moves")

# fill buffer with same pose repeated 60 times
window = [landmarks.copy() for _ in range(60)]
result = engine.compare(window)

print(f"Best match:   {result.move_name}")
print(f"DTW distance: {result.distance:.2f}")
print(f"Threshold:    {result.threshold:.2f}")
print(f"Aligned:      {result.aligned}")

score = scorer.score(result, fake_frame)
print(f"Score result: {score}")