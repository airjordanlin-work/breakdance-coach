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
  const [fetching, setFetching] = useState(true);
  const [error,    setError]    = useState(null);

  useEffect(() => {
    fetch("http://localhost:8000/moves")
      .then((r) => r.json())
      .then((d) => {
        setMoves(d.moves);
        if (d.moves.length) setSelected(d.moves[0].id);
      })
      .catch(() => setError("Cannot connect to backend — is uvicorn running?"))
      .finally(() => setFetching(false));
  }, []);

  const handleStart = async () => {
    setLoading(true);
    await onStart({ move_id: selected });
    setLoading(false);
  };
  if (fetching) return (
    <div style={{ minHeight:"100vh", background:"#060608",
      fontFamily:"'Anton','Arial Black',sans-serif",
      display:"flex", alignItems:"center", justifyContent:"center",
      flexDirection:"column", gap:16 }}>
      <div style={{ fontSize:11, letterSpacing:"0.3em", color:"#ff2d2d" }}>
        BREAKDANCE COACH
      </div>
      <div style={{ fontSize:13, letterSpacing:"0.2em", color:"#333" }}>
        LOADING MOVES...
      </div>
    </div>
  );
  
  if (error) return (
    <div style={{ minHeight:"100vh", background:"#060608",
      fontFamily:"'Anton','Arial Black',sans-serif",
      display:"flex", alignItems:"center", justifyContent:"center",
      flexDirection:"column", gap:16 }}>
      <div style={{ fontSize:11, letterSpacing:"0.3em", color:"#ff2d2d" }}>
        CONNECTION ERROR
      </div>
      <div style={{ fontSize:13, letterSpacing:"0.2em", color:"#555", maxWidth:400, textAlign:"center" }}>
        {error}
      </div>
    </div>
  );
  
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
