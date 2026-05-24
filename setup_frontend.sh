#!/bin/bash
set -e

ROOT="/Users/jordanlin/breakdance-coach/breakdance-coach"
cd "$ROOT"

echo "Creating directory structure..."
mkdir -p backend/api
mkdir -p frontend/src/pages
mkdir -p frontend/src/components
mkdir -p frontend/src/hooks

echo "Copying backend files..."
cp /dev/stdin backend/api/__init__.py << 'EOF'
EOF

echo "Copying frontend files..."

# ── index.html ────────────────────────────────────────────────────────────────
cat > frontend/index.html << 'EOF'
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Breakdance Coach</title>
    <link rel="preconnect" href="https://fonts.googleapis.com" />
    <link href="https://fonts.googleapis.com/css2?family=Anton&display=swap" rel="stylesheet" />
    <style>
      *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
      body { background: #060608; color: #fff; overflow-x: hidden; }
      button { font-family: 'Anton', 'Arial Black', sans-serif; }
    </style>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.jsx"></script>
  </body>
</html>
EOF

# ── vite.config.js ────────────────────────────────────────────────────────────
cat > frontend/vite.config.js << 'EOF'
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: { "/api": "http://localhost:8000" },
  },
});
EOF

# ── package.json ──────────────────────────────────────────────────────────────
cat > frontend/package.json << 'EOF'
{
  "name": "breakdance-coach-frontend",
  "private": true,
  "version": "0.1.0",
  "type": "module",
  "scripts": {
    "dev":     "vite",
    "build":   "vite build",
    "preview": "vite preview"
  },
  "dependencies": {
    "react":     "^18.2.0",
    "react-dom": "^18.2.0"
  },
  "devDependencies": {
    "@vitejs/plugin-react": "^4.2.0",
    "vite":                 "^5.0.0"
  }
}
EOF

# ── src/main.jsx ──────────────────────────────────────────────────────────────
cat > frontend/src/main.jsx << 'EOF'
import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
EOF

# ── src/App.jsx ───────────────────────────────────────────────────────────────
cat > frontend/src/App.jsx << 'EOF'
import { useState } from "react";
import Landing from "./pages/Landing";
import Session from "./pages/Session";
import { useSession } from "./hooks/useSession";

export default function App() {
  const { sessionId, startSession, endSession } = useSession();
  return sessionId
    ? <Session sessionId={sessionId} onEnd={endSession} />
    : <Landing onStart={startSession} />;
}
EOF

# ── hooks ─────────────────────────────────────────────────────────────────────
cat > frontend/src/hooks/useWebSocket.js << 'EOF'
import { useRef, useCallback, useEffect } from "react";

export function useWebSocket(sessionId, onMessage) {
  const ws = useRef(null);
  useEffect(() => {
    if (!sessionId) return;
    ws.current = new WebSocket(`ws://localhost:8000/ws/${sessionId}`);
    ws.current.onmessage = (e) => onMessage(JSON.parse(e.data));
    ws.current.onerror   = (e) => console.error("WS error", e);
    return () => ws.current?.close();
  }, [sessionId]);
  const sendFrame = useCallback((b64) => {
    if (ws.current?.readyState === WebSocket.OPEN)
      ws.current.send(JSON.stringify({ type: "frame", data: b64 }));
  }, []);
  return { sendFrame };
}
EOF

cat > frontend/src/hooks/useSession.js << 'EOF'
import { useState, useCallback } from "react";

export function useSession() {
  const [sessionId, setSessionId] = useState(null);
  const [loading, setLoading]     = useState(false);
  const startSession = useCallback(async (config = {}) => {
    setLoading(true);
    try {
      const res = await fetch("http://localhost:8000/session/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(config),
      });
      const { session_id } = await res.json();
      setSessionId(session_id);
      return session_id;
    } finally { setLoading(false); }
  }, []);
  const endSession = useCallback(() => setSessionId(null), []);
  return { sessionId, loading, startSession, endSession };
}
EOF

cat > frontend/src/hooks/useCamera.js << 'EOF'
import { useRef, useEffect, useCallback } from "react";

export function useCamera(onFrame, fps = 15) {
  const videoRef    = useRef(null);
  const canvasRef   = useRef(null);
  const intervalRef = useRef(null);
  useEffect(() => {
    navigator.mediaDevices.getUserMedia({ video: true }).then((stream) => {
      if (videoRef.current) videoRef.current.srcObject = stream;
    });
    return () => {
      clearInterval(intervalRef.current);
      if (videoRef.current?.srcObject)
        videoRef.current.srcObject.getTracks().forEach((t) => t.stop());
    };
  }, []);
  const startCapture = useCallback(() => {
    intervalRef.current = setInterval(() => {
      const video = videoRef.current, canvas = canvasRef.current;
      if (!video || !canvas) return;
      const ctx = canvas.getContext("2d");
      canvas.width = video.videoWidth; canvas.height = video.videoHeight;
      ctx.drawImage(video, 0, 0);
      onFrame(canvas.toDataURL("image/jpeg", 0.7).split(",")[1]);
    }, 1000 / fps);
  }, [onFrame, fps]);
  const stopCapture = useCallback(() => clearInterval(intervalRef.current), []);
  return { videoRef, canvasRef, startCapture, stopCapture };
}
EOF

# ── pages ─────────────────────────────────────────────────────────────────────
cat > frontend/src/pages/Landing.jsx << 'EOF'
import { useState, useEffect } from "react";

const DIFF_COLOR = {
  Beginner:     { bg: "#0a1f0a", border: "#1a5c1a", text: "#4cff4c" },
  Intermediate: { bg: "#1f1500", border: "#5c3d00", text: "#ffc14c" },
  Advanced:     { bg: "#1f0000", border: "#5c0000", text: "#ff4c4c" },
};

export default function Landing({ onStart }) {
  const [moves,    setMoves]    = useState([]);
  const [selected, setSelected] = useState(null);
  const [loading,  setLoading]  = useState(false);

  useEffect(() => {
    fetch("http://localhost:8000/moves")
      .then((r) => r.json())
      .then((d) => { setMoves(d.moves); if (d.moves.length) setSelected(d.moves[0].id); });
  }, []);

  const handleStart = async () => {
    setLoading(true);
    await onStart({ move_id: selected });
    setLoading(false);
  };

  return (
    <div style={{ minHeight:"100vh", background:"#060608",
      fontFamily:"'Anton','Arial Black',sans-serif", color:"#fff" }}>
      <div style={{ borderBottom:"3px solid #ff2d2d", padding:"24px 40px",
        display:"flex", justifyContent:"space-between", alignItems:"center" }}>
        <div>
          <div style={{ fontSize:11, letterSpacing:"0.3em", color:"#ff2d2d", marginBottom:4 }}>AI POWERED</div>
          <div style={{ fontSize:36, letterSpacing:"0.05em", lineHeight:1 }}>BREAKDANCE COACH</div>
        </div>
        <div style={{ textAlign:"right" }}>
          <div style={{ fontSize:10, color:"#444", letterSpacing:"0.2em" }}>POWERED BY</div>
          <div style={{ fontSize:13, color:"#666", letterSpacing:"0.1em" }}>AIST++ · MEDIAPIPE · DTW</div>
        </div>
      </div>

      <div style={{ padding:"40px", maxWidth:1200, margin:"0 auto" }}>
        <div style={{ fontSize:11, letterSpacing:"0.3em", color:"#555", marginBottom:20 }}>
          SELECT YOUR MOVE — {moves.length} AVAILABLE
        </div>

        <div style={{ display:"grid", gridTemplateColumns:"repeat(auto-fill,minmax(200px,1fr))",
          gap:12, marginBottom:48 }}>
          {moves.map((m) => {
            const dc  = DIFF_COLOR[m.difficulty] || DIFF_COLOR.Advanced;
            const sel = selected === m.id;
            return (
              <div key={m.id} onClick={() => setSelected(m.id)} style={{
                background: sel ? "#0d0d0d" : "#080808",
                border: `2px solid ${sel ? "#ff2d2d" : "#1a1a1a"}`,
                padding:"20px 16px", cursor:"pointer", position:"relative",
                transition:"border-color 0.15s",
              }}>
                {sel && <div style={{ position:"absolute", top:0, right:0,
                  background:"#ff2d2d", color:"#000", fontSize:9, fontWeight:700,
                  padding:"3px 8px", letterSpacing:"0.15em" }}>SELECTED</div>}
                <div style={{ display:"inline-block", background:dc.bg,
                  border:`1px solid ${dc.border}`, color:dc.text,
                  fontSize:9, padding:"2px 8px", letterSpacing:"0.15em", marginBottom:10 }}>
                  {m.difficulty.toUpperCase()}
                </div>
                <div style={{ fontSize:18, letterSpacing:"0.05em", marginBottom:4 }}>{m.name}</div>
                <div style={{ fontSize:10, color:"#444", letterSpacing:"0.1em" }}>
                  {m.source === "AIST++" ? "PROFESSIONAL REFERENCE" : "CUSTOM REFERENCE"}
                </div>
              </div>
            );
          })}
        </div>

        <div style={{ textAlign:"center" }}>
          <button onClick={handleStart} disabled={!selected || loading} style={{
            background: loading ? "#111" : "#ff2d2d",
            color: loading ? "#444" : "#000", border:"none",
            padding:"18px 80px", fontSize:20,
            fontFamily:"'Anton','Arial Black',sans-serif",
            letterSpacing:"0.15em",
            cursor: loading ? "not-allowed" : "pointer",
          }}>
            {loading ? "LOADING..." : "START SESSION"}
          </button>
          <div style={{ marginTop:12, fontSize:10, color:"#333", letterSpacing:"0.15em" }}>
            MAKE SURE YOUR FULL BODY IS VISIBLE
          </div>
        </div>
      </div>
    </div>
  );
}
EOF

cat > frontend/src/pages/Session.jsx << 'EOF'
import { useState, useEffect, useCallback } from "react";
import { useWebSocket } from "../hooks/useWebSocket";
import { useCamera }    from "../hooks/useCamera";
import ScoreHUD         from "../components/ScoreHUD";
import KeyframeFlash    from "../components/KeyframeFlash";

export default function Session({ sessionId, onEnd }) {
  const [frameData,   setFrameData]   = useState(null);
  const [stats,       setStats]       = useState(null);
  const [scoreResult, setScoreResult] = useState(null);
  const [fillRatio,   setFillRatio]   = useState(0);
  const [aligned,     setAligned]     = useState(false);
  const [moveName,    setMoveName]    = useState("");

  const handleMessage = useCallback((msg) => {
    if (msg.frame)                    setFrameData(`data:image/jpeg;base64,${msg.frame}`);
    if (msg.stats)                    setStats(msg.stats);
    if (msg.fill_ratio !== undefined) setFillRatio(msg.fill_ratio);
    if (msg.aligned   !== undefined)  setAligned(msg.aligned);
    if (msg.move_name)                setMoveName(msg.move_name);
    if (msg.score_result)             setScoreResult(msg.score_result);
  }, []);

  const { sendFrame }                              = useWebSocket(sessionId, handleMessage);
  const { videoRef, canvasRef, startCapture, stopCapture } = useCamera(sendFrame, 15);

  useEffect(() => { startCapture(); return () => stopCapture(); }, [startCapture, stopCapture]);

  return (
    <div style={{ minHeight:"100vh", background:"#060608",
      fontFamily:"'Anton','Arial Black',sans-serif", color:"#fff",
      display:"flex", flexDirection:"column" }}>

      <div style={{ borderBottom:"2px solid #1a1a1a", padding:"14px 24px",
        display:"flex", justifyContent:"space-between", alignItems:"center" }}>
        <div style={{ fontSize:14, letterSpacing:"0.15em", color:"#ff2d2d" }}>BREAKDANCE COACH</div>
        <div style={{ fontSize:11, letterSpacing:"0.2em",
          color: aligned ? "#4cff4c" : "#555",
          border:`1px solid ${aligned ? "#4cff4c" : "#222"}`, padding:"4px 12px" }}>
          {aligned ? "ALIGNED" : moveName ? `MOVE: ${moveName.toUpperCase()}` : "DETECTING..."}
        </div>
        <button onClick={onEnd} style={{ background:"transparent", border:"1px solid #333",
          color:"#555", padding:"6px 16px", fontFamily:"'Anton',sans-serif",
          fontSize:11, letterSpacing:"0.15em", cursor:"pointer" }}>
          END SESSION
        </button>
      </div>

      <div style={{ flex:1, display:"flex", overflow:"hidden" }}>
        <div style={{ flex:1, position:"relative", background:"#000" }}>
          {frameData
            ? <img src={frameData} alt="live" style={{ width:"100%", height:"100%", objectFit:"contain", display:"block" }} />
            : <div style={{ width:"100%", height:"100%", display:"flex", alignItems:"center",
                justifyContent:"center", color:"#333", fontSize:13, letterSpacing:"0.2em" }}>
                CONNECTING...
              </div>
          }
          <div style={{ position:"absolute", bottom:0, left:0, right:0, height:4, background:"#111" }}>
            <div style={{ height:"100%", width:`${fillRatio * 100}%`,
              background: fillRatio >= 1 ? "#4cff4c" : "#ff2d2d", transition:"width 0.3s" }} />
          </div>
          {scoreResult && <KeyframeFlash result={scoreResult} onDone={() => setScoreResult(null)} />}
        </div>

        <div style={{ width:220, background:"#080808", borderLeft:"2px solid #111", padding:20 }}>
          {stats && <ScoreHUD stats={stats} />}
          <div style={{ marginTop:24 }}>
            <div style={{ fontSize:9, letterSpacing:"0.2em", color:"#333", marginBottom:8 }}>BUFFER</div>
            <div style={{ fontSize:22, color: fillRatio >= 1 ? "#4cff4c" : "#ff2d2d" }}>
              {fillRatio >= 1 ? "READY" : `${Math.round(fillRatio * 100)}%`}
            </div>
          </div>
        </div>
      </div>

      <video ref={videoRef} autoPlay playsInline muted style={{ display:"none" }} />
      <canvas ref={canvasRef} style={{ display:"none" }} />
    </div>
  );
}
EOF

# ── components ────────────────────────────────────────────────────────────────
cat > frontend/src/components/ScoreHUD.jsx << 'EOF'
export default function ScoreHUD({ stats }) {
  const gc = { S:"#ffd700", A:"#4cff4c", B:"#4ccfff", C:"#ffc14c", D:"#ff4c4c" }[stats.grade] || "#fff";
  const total = Math.max(stats.perfects + stats.closes + stats.misses, 1);
  return (
    <div>
      <div style={{ fontSize:9, letterSpacing:"0.2em", color:"#333", marginBottom:8 }}>SCORE</div>
      <div style={{ fontSize:48, lineHeight:1, marginBottom:4 }}>{stats.total_points.toLocaleString()}</div>
      <div style={{ display:"inline-block", border:`2px solid ${gc}`, color:gc,
        fontSize:22, padding:"2px 16px", marginBottom:20 }}>{stats.grade}</div>
      <div style={{ fontSize:9, letterSpacing:"0.2em", color:"#333", marginBottom:8 }}>STREAK</div>
      <div style={{ fontSize:28, marginBottom:20, color: stats.streak > 2 ? "#ffd700" : "#fff" }}>{stats.streak}×</div>
      <div style={{ fontSize:9, letterSpacing:"0.2em", color:"#333", marginBottom:10 }}>BREAKDOWN</div>
      <div style={{ display:"flex", gap:8 }}>
        {[["P", stats.perfects, "#4cff4c"], ["C", stats.closes, "#ffc14c"], ["M", stats.misses, "#ff4c4c"]].map(([l,v,c]) => (
          <div key={l} style={{ flex:1, background:"#0d0d0d", border:"1px solid #1a1a1a", padding:"8px 0", textAlign:"center" }}>
            <div style={{ fontSize:18, color:c }}>{v}</div>
            <div style={{ fontSize:9, color:"#444", letterSpacing:"0.1em" }}>{l}</div>
          </div>
        ))}
      </div>
      <div style={{ marginTop:12, height:3, background:"#111", display:"flex" }}>
        {[[stats.perfects/total,"#4cff4c"],[stats.closes/total,"#ffc14c"],[stats.misses/total,"#ff4c4c"]].map(([w,c],i) => (
          <div key={i} style={{ width:`${w*100}%`, background:c }} />
        ))}
      </div>
    </div>
  );
}
EOF

cat > frontend/src/components/KeyframeFlash.jsx << 'EOF'
import { useEffect } from "react";
const TC = { Perfect:"#4cff4c", Close:"#ffc14c", Miss:"#ff4c4c" };
export default function KeyframeFlash({ result, onDone }) {
  useEffect(() => { const t = setTimeout(onDone, 2500); return () => clearTimeout(t); }, [result, onDone]);
  if (!result) return null;
  return (
    <div style={{ position:"absolute", top:20, right:20, background:"rgba(6,6,8,0.92)",
      border:"1px solid #222", padding:"16px 20px", minWidth:200 }}>
      <div style={{ fontSize:9, letterSpacing:"0.2em", color:"#555", marginBottom:8 }}>KEYFRAME</div>
      <div style={{ fontSize:28, marginBottom:12 }}>+{result.points} PTS</div>
      {result.results.map((r, i) => (
        <div key={i} style={{ marginBottom:8 }}>
          <div style={{ display:"flex", justifyContent:"space-between",
            fontSize:11, letterSpacing:"0.1em", marginBottom:4 }}>
            <span style={{ color:"#888" }}>{r.joint.toUpperCase()}</span>
            <span style={{ color:TC[r.tier] }}>{r.tier.toUpperCase()}</span>
          </div>
          <div style={{ height:2, background:"#111" }}>
            <div style={{ height:"100%", width:`${Math.max(0,(1-r.diff/90)*100)}%`,
              background:TC[r.tier], transition:"width 0.4s" }} />
          </div>
          <div style={{ fontSize:9, color:"#444", marginTop:2 }}>{r.diff} DEG OFF</div>
        </div>
      ))}
    </div>
  );
}
EOF

# ── update backend requirements ───────────────────────────────────────────────
cat > backend/requirements.txt << 'EOF'
mediapipe
opencv-python
numpy
fastdtw
scipy
kokoro-onnx
soundfile
python-dotenv
fastapi
uvicorn[standard]
websockets
python-multipart
EOF

# ── backend api __init__ ──────────────────────────────────────────────────────
touch backend/api/__init__.py

echo ""
echo "All files created successfully!"
echo ""
echo "Next steps:"
echo "  1. source backend/.venv/bin/activate"
echo "  2. pip install fastapi 'uvicorn[standard]' websockets python-multipart"
echo "  3. cd frontend && npm install"
echo "  4. Terminal 1: cd backend && uvicorn api.server:app --reload --port 8000"
echo "  5. Terminal 2: cd frontend && npm run dev"
echo "  6. Open http://localhost:5173"