# Breakdance Coach — Real-Time AI Movement Coaching

A computer vision app that tracks your skeletal keypoints via webcam and gives you real-time feedback on breakdancing move accuracy. Built with MediaPipe, OpenCV, and Dynamic Time Warping.

[View Architecture Diagram](https://airjordanlin-work.github.io/breakdance-coach/architecture.html)

---

## Demo

> Drop a GIF here once you have the overlay running — e.g. `![Demo](demo/demo.gif)`

---

## How it works

The app captures your webcam feed frame-by-frame and passes each frame through a MediaPipe pose estimator, extracting 33 skeletal landmarks (x, y, z, visibility). Those keypoints are normalized to a hip-center coordinate system so the comparison is body-size agnostic, then buffered into a 60-frame sliding window.

That window is compared against a pre-recorded reference move using Dynamic Time Warping (DTW), which handles the fact that two people execute the same move at different speeds. At the peak of each move, a keyframe scorer checks your joint angles against target angles — if you're within the threshold, you score points. A combo multiplier rewards consecutive clean hits.

The full system architecture is documented in [`docs/architecture.html`](docs/architecture.html).

---

## Features

- Real-time 33-point skeletal tracking via MediaPipe
- Body-size-agnostic keypoint normalization (hip-center origin, torso scale)
- DTW-based sequence comparison against reference moves (runs async, no frame drops)
- Keyframe thresholding scorer with per-joint angle feedback
- Combo multiplier and rolling score window
- OpenCV overlay with live HUD — score, grade, and joint-level correction hints
- Fallback to sample video if no webcam is present (`--source demo/sample.mp4`)
- Docker support — runs on any laptop with a single command

---

## Tech stack

| Layer | Library |
|---|---|
| Pose estimation | `mediapipe` |
| Frame capture & overlay | `opencv-python` |
| Keypoint math | `numpy` |
| DTW comparison | `fastdtw` |
| Async pipeline | `asyncio`, `concurrent.futures` |
| Packaging | `docker` |

---

## Quickstart

### Run with Docker (recommended)

```bash
git clone https://github.com/YOUR_USERNAME/breakdance-coach.git
cd breakdance-coach
docker compose up
```

This runs the app against the included sample video. To use your webcam instead:

```bash
docker compose run app python app/main.py --source webcam
```

### Run locally

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app/main.py --source demo/sample.mp4
```

Python 3.10+ required.

---

## Project structure

```
breakdance-coach/
├── app/
│   ├── main.py              # entry point, webcam loop
│   ├── pose_estimator.py    # MediaPipe wrapper + normalization
│   ├── dtw_engine.py        # async DTW comparison
│   ├── scorer.py            # keyframe thresholding + combo logic
│   └── overlay.py           # OpenCV HUD renderer
├── reference_moves/
│   ├── windmill.npy         # reference keypoint sequences
│   └── windmill_meta.json   # keyframe timestamps + target angles
├── docs/
│   └── architecture.html    # full system architecture diagram
├── demo/
│   └── sample.mp4           # sample video for recruiter demo
├── tests/
│   └── test_scorer.py
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Adding a new reference move

1. Record yourself performing the move cleanly — any webcam works.
2. Run the capture script to extract and save the keypoint sequence:
   ```bash
   python app/main.py --record --output reference_moves/your_move_name
   ```
3. Edit the generated `your_move_name_meta.json` to set keyframe timestamps and target joint angles.
4. The move will appear automatically in the comparison engine on next run.

---

## Data engineering notes

The skeletal keypoint stream is treated as a multivariate time series — 33 landmarks × 3 spatial dimensions × N frames. Key pipeline decisions:

- **Normalization** — all coordinates translated to hip midpoint origin and scaled by torso length, making comparisons body-size invariant.
- **Circular buffer** — `collections.deque(maxlen=60)` provides O(1) append/pop, keeping the sliding window lock-free on the main render thread.
- **Async DTW** — comparison runs in a `ThreadPoolExecutor` worker every 3 frames, decoupled from the 30 fps OpenCV render loop to avoid blocking.
- **Reference storage** — sequences stored as binary `.npy` arrays for fast `numpy` load; metadata (move name, keyframe timestamps, target angles, difficulty) stored as plain JSON alongside.

---

## Resume / skills demonstrated

`Python` · `MediaPipe` · `OpenCV` · `NumPy` · `Dynamic Time Warping` · `Time-series data pipelines` · `Real-time stream processing` · `asyncio` · `Docker` · `Computer Vision`

---

## Roadmap

- [ ] Move recognition via embedding vectors + FAISS nearest-neighbor search
- [ ] Web UI (FastAPI + WebSocket stream)
- [ ] Leaderboard / session history with SQLite
- [ ] Mobile support via TensorFlow Lite MoveNet

---

## License

MIT
