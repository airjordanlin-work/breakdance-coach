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
