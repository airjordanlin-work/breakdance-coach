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
      const video  = videoRef.current;
      const canvas = canvasRef.current;
      if (!video || !canvas || !video.videoWidth) return;
      const ctx = canvas.getContext("2d");
      canvas.width  = video.videoWidth;
      canvas.height = video.videoHeight;
      ctx.drawImage(video, 0, 0);
      onFrame(canvas.toDataURL("image/jpeg", 0.7).split(",")[1]);
    }, 1000 / fps);
  }, [onFrame, fps]);

  const stopCapture = useCallback(() => clearInterval(intervalRef.current), []);
  return { videoRef, canvasRef, startCapture, stopCapture };
}