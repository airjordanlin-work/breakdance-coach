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
