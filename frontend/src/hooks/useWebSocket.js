import { useRef, useCallback, useEffect } from "react";

export function useWebSocket(sessionId, onMessage, onOpen) {
  const ws      = useRef(null);
  const onOpenRef = useRef(onOpen);
  const onMsgRef  = useRef(onMessage);

  useEffect(() => { onOpenRef.current = onOpen; }, [onOpen]);
  useEffect(() => { onMsgRef.current  = onMessage; }, [onMessage]);

  useEffect(() => {
    if (!sessionId) return;
    ws.current = new WebSocket(`ws://localhost:8000/ws/${sessionId}`);
    ws.current.onopen    = () => onOpenRef.current?.();
    ws.current.onmessage = (e) => onMsgRef.current?.(JSON.parse(e.data));
    ws.current.onerror   = (e) => console.error("WS error", e);
    return () => ws.current?.close();
  }, [sessionId]);

  const sendFrame = useCallback((b64) => {
    if (ws.current?.readyState === WebSocket.OPEN)
      ws.current.send(JSON.stringify({ type: "frame", data: b64 }));
  }, []);

  return { sendFrame };
}