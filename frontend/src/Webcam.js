
import { useEffect, useRef, useState } from "react";
import Overlay from "./Overlay";

const VIDEO_WIDTH  = 640;
const VIDEO_HEIGHT = 480;

const STILL_THRESHOLD = 0.022; // max avg landmark movement to be considered still
const STILL_FRAMES    = 4;     // consecutive still frames required

export default function Webcam({ onLandmarks, onNoHand }) {
  const videoRef      = useRef(null);
  const handsRef      = useRef(null);
  const camRef        = useRef(null);
  const frameSkip     = useRef(0);
  const prevLandmarks = useRef(null);  // previous frame landmarks
  const stillCount    = useRef(0);     // consecutive still frame counter

  const [overlayLandmarks, setOverlayLandmarks] = useState([]);
  const [status, setStatus] = useState("loading"); // loading | ready | error

  // ── Initialize MediaPipe Hands ────────────────────────────────────────────
  useEffect(() => {
    let active = true;

    async function init() {
      try {
        // Dynamic import so webpack bundles it correctly
        const { Hands } = await import("@mediapipe/hands");
        const { Camera } = await import("@mediapipe/camera_utils");

        const hands = new Hands({
          locateFile: (file) =>
            `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`,
        });

        hands.setOptions({
          maxNumHands: 1,
          modelComplexity: 0,
          minDetectionConfidence: 0.5,
          minTrackingConfidence: 0.5,
        });

        hands.onResults((results) => {
          if (!active) return;
          if (
            results.multiHandLandmarks &&
            results.multiHandLandmarks.length > 0
          ) {
            const lm = results.multiHandLandmarks[0]; // first hand only
            setOverlayLandmarks(lm);

            // Flatten to 63 floats: [x0,y0,z0, x1,y1,z1, ..., x20,y20,z20]
            const flat = lm.flatMap((p) => [p.x, p.y, p.z]);

            // ── Hand stillness check ──────────────────────────────────────
            let isStill = false;
            if (prevLandmarks.current) {
              const movement = lm.reduce((sum, pt, i) => {
                const prev = prevLandmarks.current[i];
                return sum + Math.abs(pt.x - prev.x) + Math.abs(pt.y - prev.y);
              }, 0) / lm.length;
              if (movement < STILL_THRESHOLD) {
                stillCount.current = Math.min(stillCount.current + 1, STILL_FRAMES);
              } else {
                stillCount.current = 0;
              }
              isStill = stillCount.current >= STILL_FRAMES;
            }
            prevLandmarks.current = lm.map(p => ({ x: p.x, y: p.y }));

            onLandmarks(flat, isStill);
          } else {
            setOverlayLandmarks([]);
            onNoHand();
          }
        });

        const cam = new Camera(videoRef.current, {
          onFrame: async () => {
            await hands.send({ image: videoRef.current });
          },
          width: VIDEO_WIDTH,
          height: VIDEO_HEIGHT,
        });

        await cam.start();
        camRef.current = cam;
        handsRef.current = hands;
        if (active) setStatus("ready");
      } catch (err) {
        console.error("MediaPipe init failed:", err);
        if (active) setStatus("error");
      }
    }

    init();
    return () => {
      active = false;
      if (camRef.current) {
        camRef.current.stop();
      }
      if (handsRef.current) {
        handsRef.current.close();
      }
    };
  }, [onLandmarks, onNoHand]);

  return (
    <div
      style={{
        position: "relative",
        width: VIDEO_WIDTH,
        height: VIDEO_HEIGHT,
        borderRadius: 16,
        overflow: "hidden",
        boxShadow: "0 0 40px rgba(0,212,255,0.2)",
        border: "1px solid rgba(0,212,255,0.2)",
      }}
    >
      {/* Status overlay */}
      {status === "loading" && (
        <div className="cam-status">
          <div className="cam-spinner" />
          <span>Initializing Camera…</span>
        </div>
      )}
      {status === "error" && (
        <div className="cam-status cam-error">
          <span>⚠ Camera / MediaPipe error</span>
          <small>Check console for details</small>
        </div>
      )}

      {/* Video element */}
      <video
        ref={videoRef}
        autoPlay
        playsInline
        muted
        width={VIDEO_WIDTH}
        height={VIDEO_HEIGHT}
        style={{
          display: "block",
          transform: "scaleX(-1)", // mirror for natural feel
          width: "100%",
          height: "100%",
          objectFit: "cover",
        }}
      />

      {/* Skeleton overlay */}
      <Overlay
        landmarks={overlayLandmarks}
        width={VIDEO_WIDTH}
        height={VIDEO_HEIGHT}
      />

      {/* Corner scan lines decoration */}
      <div className="corner-tl" />
      <div className="corner-tr" />
      <div className="corner-bl" />
      <div className="corner-br" />
    </div>
  );
}
