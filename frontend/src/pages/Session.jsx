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
