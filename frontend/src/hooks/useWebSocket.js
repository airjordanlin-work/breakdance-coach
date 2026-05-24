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
