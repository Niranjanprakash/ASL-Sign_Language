/**
 * Overlay.js - Canvas-based hand skeleton renderer.
 * Draws 21 green landmark dots + blue connection lines on top of the webcam feed.
 */

import { useEffect, useRef } from "react";

/* MediaPipe hand connections (pairs of landmark indices) */
const CONNECTIONS = [
  [0,1],[1,2],[2,3],[3,4],
  [0,5],[5,6],[6,7],[7,8],
  [0,9],[9,10],[10,11],[11,12],
  [0,13],[13,14],[14,15],[15,16],
  [0,17],[17,18],[18,19],[19,20],
  [5,9],[9,13],[13,17],
];

export default function Overlay({ landmarks, width, height }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, width, height);

    if (!landmarks || landmarks.length === 0) return;

    // Scale factor: drawing coords are 640×480 but canvas may be rendered smaller
    const scaleX = width  / (canvas.offsetWidth  || width);
    const scaleY = height / (canvas.offsetHeight || height);
    const scale  = Math.min(scaleX, scaleY);          // uniform scale
    const dotTip  = Math.max(2,  3  / scale);          // fingertip dot radius
    const dotJoint= Math.max(1.5, 2 / scale);          // joint dot radius
    const dotCore = Math.max(0.5, 1 / scale);          // white core radius
    const lineW   = Math.max(1,   1.5 / scale);        // connection line width

    // Convert normalized [0,1] coords → pixel coords
    const pts = landmarks.map((lm) => ({
      x: lm.x * width,
      y: lm.y * height,
    }));

    // Draw connections
    ctx.strokeStyle = "rgba(99, 102, 241, 0.75)";
    ctx.lineWidth = lineW;
    ctx.shadowColor = "rgba(99, 102, 241, 0.5)";
    ctx.shadowBlur = 4 / scale;
    for (const [a, b] of CONNECTIONS) {
      ctx.beginPath();
      ctx.moveTo(pts[a].x, pts[a].y);
      ctx.lineTo(pts[b].x, pts[b].y);
      ctx.stroke();
    }

    // Draw landmarks
    ctx.shadowColor = "rgba(139, 92, 246, 0.8)";
    ctx.shadowBlur = 5 / scale;
    pts.forEach((pt, i) => {
      const isTip = [4, 8, 12, 16, 20].includes(i);
      ctx.beginPath();
      ctx.arc(pt.x, pt.y, isTip ? dotTip : dotJoint, 0, Math.PI * 2);
      ctx.fillStyle = isTip ? "#a78bfa" : "#8b5cf6";
      ctx.fill();

      ctx.beginPath();
      ctx.arc(pt.x, pt.y, dotCore, 0, Math.PI * 2);
      ctx.fillStyle = "#ffffff";
      ctx.fill();
    });

    ctx.shadowBlur = 0;
  }, [landmarks, width, height]);

  return (
    <canvas
      ref={canvasRef}
      width={width}
      height={height}
      style={{
        position: "absolute",
        top: 0,
        left: 0,
        width: "100%",
        height: "100%",
        pointerEvents: "none",
        zIndex: 10,
        transform: "scaleX(-1)",
      }}
    />
  );
}
