"""
FastAPI backend — WebSocket streaming for Breakdance Coach.

Endpoints:
    GET  /health
    GET  /moves
    POST /session/start
    WS   /ws/{session_id}
"""

from __future__ import annotations

import base64
import json
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import sys
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.pose_estimator import PoseEstimator, is_pose_reliable
from app.buffer import PoseBuffer, WINDOW_LEN
from app.dtw_engine import DTWEngine
from app.scorer import Scorer
from app import overlay

app = FastAPI(title="Breakdance Coach API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

REFERENCE_DIR      = Path(__file__).resolve().parent.parent / "reference_moves"
KEYFRAME_FLASH     = 30
_sessions: dict[str, "CoachingSession"] = {}
_executor          = ThreadPoolExecutor(max_workers=4)


class SessionConfig(BaseModel):
    voice_gender: str  = "female"
    zoom: float        = 1.0


class CoachingSession:
    def __init__(self, config: SessionConfig) -> None:
        self.estimator     = PoseEstimator(model_complexity=0)
        self.buf           = PoseBuffer()
        self.engine        = DTWEngine(REFERENCE_DIR)
        self.scorer        = Scorer(REFERENCE_DIR)
        self.future        = None
        self.flash_counter = 0
        self.score_result  = None
        self.dtw_result    = None
        self.pose_frame    = None
        self.frame_idx     = 0
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        self.estimator.process(dummy)

    def process_frame(self, b64: str) -> dict:
        """Synchronous frame processing — runs in thread pool."""
        data  = base64.b64decode(b64)
        arr   = np.frombuffer(data, np.uint8)
        bgr   = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        frame = cv2.flip(bgr, 1)
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose  = self.estimator.process(rgb, keep_raw=True)

        if pose is not None and is_pose_reliable(pose):
            self.pose_frame = pose
            self.buf.add(pose)

        if self.buf.is_ready() and self.future is None:
            window      = [f.landmarks for f in self.buf._frames]
            self.future = self.engine.compare_async(window)

        if self.future is not None and self.future.done():
            try:
                self.dtw_result = self.future.result()
            except Exception:
                self.dtw_result = None
            self.future = None

            if self.dtw_result and self.dtw_result.aligned and self.pose_frame:
                s = self.scorer.score(self.dtw_result, self.pose_frame)
                if s is not None:
                    self.score_result  = s
                    self.flash_counter = KEYFRAME_FLASH

        fill_ratio = len(self.buf) / WINDOW_LEN
        move_name  = (self.dtw_result.move_name
                      if self.dtw_result else self.engine.move_name)

        annotated, self.flash_counter = overlay.render(
            frame,
            pose_frame=self.pose_frame,
            dtw_result=self.dtw_result,
            score_result=self.score_result,
            stats=self.scorer.stats(),
            fill_ratio=fill_ratio,
            flash_counter=self.flash_counter,
            move_name=move_name,
            buf_len=len(self.buf),
        )

        self.frame_idx += 1
        stats = self.scorer.stats()

        _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 70])
        frame_b64 = base64.b64encode(buf).decode()

        return {
            "frame":      frame_b64,
            "fill_ratio": round(fill_ratio, 3),
            "aligned":    bool(self.dtw_result.aligned) if self.dtw_result else False,
            "move_name":  move_name or "",
            "stats": {
                "total_points": round(stats.total_points),
                "grade":        stats.grade,
                "streak":       stats.streak,
                "perfects":     stats.perfects,
                "closes":       stats.closes,
                "misses":       stats.misses,
            },
            "score_result": {
                "points":  round(self.score_result.points_this_attempt),
                "results": [
                    {
                        "joint":  r.joint_name.replace("_", " "),
                        "tier":   r.tier,
                        "diff":   round(r.diff, 1),
                        "points": round(r.points_earned),
                    }
                    for r in self.score_result.results
                ],
            } if self.score_result and self.flash_counter > 0 else None,
        }

    def close(self) -> None:
        self.estimator.close()
        self.engine.shutdown()


def _infer_difficulty(stem: str) -> str:
    s = stem.lower()
    if any(x in s for x in ["toprock", "basic", "_00", "_01"]):
        return "Beginner"
    if any(x in s for x in ["footwork", "freeze", "_02", "_03", "_04"]):
        return "Intermediate"
    return "Advanced"


@app.get("/health")
async def health():
    return {"status": "ok", "moves": len(list(REFERENCE_DIR.glob("*.npy")))}


@app.get("/moves")
async def list_moves():
    moves = []
    for npy in sorted(REFERENCE_DIR.glob("*.npy")):
        meta_path = REFERENCE_DIR / f"{npy.stem}_meta.json"
        meta = {}
        if meta_path.exists():
            with meta_path.open() as f:
                meta = json.load(f)
        moves.append({
            "id":         npy.stem,
            "name":       meta.get("name", npy.stem).replace("_", " ").upper(),
            "source":     meta.get("source", "custom"),
            "difficulty": _infer_difficulty(npy.stem),
            "original":   meta.get("original_file", ""),
        })
    return {"moves": moves}


@app.post("/session/start")
async def start_session(config: SessionConfig):
    sid = str(uuid.uuid4())
    _sessions[sid] = CoachingSession(config)
    return {"session_id": sid}


@app.websocket("/ws/{session_id}")
async def ws_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    session = _sessions.get(session_id)
    if not session:
        await websocket.send_json({"error": "session not found"})
        await websocket.close()
        return

    import asyncio
    loop = asyncio.get_event_loop()

    try:
        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)
            if msg.get("type") == "frame":
                payload = await loop.run_in_executor(
                    _executor, session.process_frame, msg["data"]
                )
                await websocket.send_text(json.dumps(payload))
            elif msg.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
    except WebSocketDisconnect:
        pass
    finally:
        if session_id in _sessions:
            _sessions[session_id].close()
            del _sessions[session_id]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)