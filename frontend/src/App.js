import { useState, useCallback, useRef, useEffect } from "react";
import "./App.css";
import WebcamComponent from "./Webcam";
import Prediction from "./Prediction";
import { predictASL, healthCheck } from "./api";

const BUFFER_SIZE      = 5;
const HOLD_MS          = 300;
const COOLDOWN_MS      = 800;
const AUTO_SPEAK_DELAY = 2500;
const MIN_CONFIDENCE   = 0.80;  // below this → no sign detected

function majorityVote(arr) {
  if (!arr.length) return null;
  const freq = {};
  for (const v of arr) freq[v] = (freq[v] || 0) + 1;
  return Object.entries(freq).sort((a, b) => b[1] - a[1])[0][0];
}

// eslint-disable-next-line no-unused-vars
const LETTER_NAMES = {
  A:"Ay", B:"Bee", C:"See", D:"Dee", E:"Ee", F:"Ef",
  G:"Jee", H:"Aitch", I:"Eye", J:"Jay", K:"Kay", L:"El",
  M:"Em", N:"En", O:"Oh", P:"Pee", Q:"Cue", R:"Ar",
  S:"Ess", T:"Tee", U:"You", V:"Vee", W:"Double-you",
  X:"Ex", Y:"Why", Z:"Zee",
};

// ── Tick sound via AudioContext ──────────────────────────────────────────────
function playTick() {
  try {
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.connect(gain);
    gain.connect(ctx.destination);
    osc.frequency.value = 880;
    osc.type = "sine";
    gain.gain.setValueAtTime(0.15, ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.12);
    osc.start(ctx.currentTime);
    osc.stop(ctx.currentTime + 0.12);
  } catch (_) {}
}

export default function App() {
  const [result, setResult]             = useState(null);
  const [geoFeatures, setGeoFeatures]   = useState(null);
  const [noHand, setNoHand]             = useState(true);
  const [noSign, setNoSign]             = useState(false); // hand present but no valid sign
  const [backendOk, setBackend]         = useState(null);
  const [wakingUp, setWakingUp]         = useState(false);
  const [fps, setFps]                   = useState(0);
  const [bufferState, setBufferState]   = useState([]);
  const [word, setWord]                 = useState("");
  const [holdProgress, setHoldProgress] = useState(0);
  const [isSpeaking, setIsSpeaking]     = useState(false);
  const [lastAdded, setLastAdded]       = useState(null); // for flash animation

  const buffer         = useRef([]);
  const lastTime       = useRef(Date.now());
  const framesCnt      = useRef(0);
  const pendingReq     = useRef(false);
  const holdRef        = useRef(null);
  const holdStart      = useRef(null);
  const holdLetter     = useRef(null);
  const progressRef    = useRef(null);
  const cooldownMap    = useRef({});   // letter → timestamp last added
  const autoSpeakRef   = useRef(null); // auto-speak timeout
  const wordRef        = useRef("");   // mirror of word for closures
  const displayRef     = useRef(null); // word display scroll ref

  // keep wordRef in sync
  useEffect(() => { wordRef.current = word; }, [word]);

  // ── Health check ────────────────────────────────────────────────────────
  useEffect(() => {
    const timeout = setTimeout(() => setWakingUp(true), 4000); // show banner if takes >4s
    healthCheck()
      .then((r) => {
        clearTimeout(timeout);
        setWakingUp(false);
        setBackend(r.status === "ok" ? (r.model_loaded ? "online" : "missing_model") : "offline");
      })
      .catch(() => {
        clearTimeout(timeout);
        setWakingUp(false);
        setBackend("offline");
      });
    return () => clearTimeout(timeout);
  }, []);

  // ── FPS counter ─────────────────────────────────────────────────────────
  useEffect(() => {
    const id = setInterval(() => {
      const now = Date.now();
      const dt  = (now - lastTime.current) / 1000;
      setFps(Math.round(framesCnt.current / dt));
      framesCnt.current = 0;
      lastTime.current  = now;
    }, 1000);
    return () => clearInterval(id);
  }, []);

  // ── Auto-scroll word display to end ─────────────────────────────────────
  useEffect(() => {
    if (displayRef.current) {
      displayRef.current.scrollLeft = displayRef.current.scrollWidth;
    }
  }, [word]);

  // ── Speak word via Web Speech API ───────────────────────────────────────
  const speakWord = useCallback((text) => {
    if (!text?.trim() || !window.speechSynthesis) return;
    window.speechSynthesis.cancel();
    const utt = new SpeechSynthesisUtterance(text.trim().toLowerCase());
    utt.rate  = 0.9;
    utt.pitch = 1.05;
    utt.onstart = () => setIsSpeaking(true);
    utt.onend   = () => setIsSpeaking(false);
    utt.onerror = () => setIsSpeaking(false);
    window.speechSynthesis.speak(utt);
  }, []);

  // ── Reset auto-speak timer on every letter add ──────────────────────────
  const resetAutoSpeak = useCallback(() => {
    clearTimeout(autoSpeakRef.current);
    autoSpeakRef.current = setTimeout(() => {
      if (wordRef.current.trim()) speakWord(wordRef.current);
    }, AUTO_SPEAK_DELAY);
  }, [speakWord]);

  // ── Cancel ongoing hold ─────────────────────────────────────────────────
  const cancelHold = useCallback(() => {
    clearTimeout(holdRef.current);
    cancelAnimationFrame(progressRef.current);
    holdLetter.current = null;
    holdStart.current  = null;
    setHoldProgress(0);
  }, []);

  // ── Commit a letter to the word ─────────────────────────────────────────
  const commitLetter = useCallback((letter) => {
    // Cooldown check — prevent same letter spamming
    const now = Date.now();
    if (cooldownMap.current[letter] && now - cooldownMap.current[letter] < COOLDOWN_MS) return;
    cooldownMap.current[letter] = now;

    playTick();

    if (letter === "space") {
      setWord(w => w + " ");
      setLastAdded("␣");
    } else if (letter === "del") {
      setWord(w => w.slice(0, -1));
      setLastAdded("⌫");
    } else {
      const ch = letter.toUpperCase();
      setWord(w => w + ch);
      setLastAdded(ch);
    }

    // Flash reset
    setTimeout(() => setLastAdded(null), 400);
    resetAutoSpeak();
  }, [resetAutoSpeak]);

  // ── Start hold timer for a letter ───────────────────────────────────────
  const startHold = useCallback((letter) => {
    if (holdLetter.current === letter) return;
    cancelHold();
    if (letter === "nothing") return;

    holdLetter.current = letter;
    holdStart.current  = Date.now();

    const animate = () => {
      const pct = Math.min(((Date.now() - holdStart.current) / HOLD_MS) * 100, 100);
      setHoldProgress(pct);
      if (pct < 100) progressRef.current = requestAnimationFrame(animate);
    };
    progressRef.current = requestAnimationFrame(animate);

    holdRef.current = setTimeout(() => {
      setHoldProgress(0);
      const committed = holdLetter.current;
      holdLetter.current = null;
      commitLetter(committed);
    }, HOLD_MS);
  }, [cancelHold, commitLetter]);

  // ── On new landmarks from webcam ─────────────────────────────────────────
  const handleLandmarks = useCallback(async (flat63, isStill) => {
    setNoHand(false);
    framesCnt.current++;

    // If hand is moving, show no sign and reset — don't block pipeline
    if (!isStill) {
      cancelHold();
      setNoSign(true);
      setResult(null);
      buffer.current = [];
      return;
    }

    if (pendingReq.current) return;

    try {
      pendingReq.current = true;
      const res = await predictASL(flat63);

      // Gate on minimum confidence
      // nothing → always suppress
      // space/del → require higher confidence (0.92) to avoid false triggers
      // letters  → require MIN_CONFIDENCE (0.80)
      const SPECIAL_CONFIDENCE = 0.92;
      const isNothing = res.prediction === "nothing";
      const isSpecial = res.prediction === "space" || res.prediction === "del";
      const minConf   = isSpecial ? SPECIAL_CONFIDENCE : MIN_CONFIDENCE;
      if (isNothing || res.confidence < minConf) {
        cancelHold();
        setNoSign(true);
        setResult(null);
        return;
      }

      setNoSign(false);
      if (res.geometric) setGeoFeatures(res.geometric);
      buffer.current.push(res.prediction);
      if (buffer.current.length > BUFFER_SIZE) buffer.current.shift();
      const stablePred = majorityVote(buffer.current);
      setBufferState([...buffer.current]);

      const finalResult = {
        ...res,
        prediction: stablePred,
        status: stablePred === res.prediction ? res.status : "uncertain",
      };

      setResult(finalResult);

      if (finalResult.status === "stable") {
        startHold(stablePred);
      } else {
        cancelHold();
      }
    } catch (err) {
      console.warn("Predict error:", err.message);
    } finally {
      pendingReq.current = false;
    }
  }, [startHold, cancelHold]);

  const handleNoHand = useCallback(() => {
    setNoHand(true);
    setNoSign(false);
    buffer.current = [];
    setResult(null);
    setGeoFeatures(null);
    cancelHold();
  }, [cancelHold]);

  return (
    <div className="app">
      {/* ── Waking up banner ── */}
      {wakingUp && backendOk === null && (
        <div className="wakeup-banner">
          <span className="wakeup-spinner" />
          Backend is waking up on Render free tier — this may take up to 60 seconds on first load…
        </div>
      )}

      {/* ── Header ── */}
      <header className="app-header">
        <div className="header-left">
          <div className="logo-glyph">ASL</div>
          <div>
            <h1 className="app-title">Confusion-Aware ASL Recognition</h1>
            <p className="app-sub">Real-Time Hand Sign Detection · MLP Neural Network</p>
          </div>
        </div>
        <div className="header-right">
          <div className="status-pill">
            <div className={`status-dot ${backendOk === "online" ? "online" : backendOk === "missing_model" ? "missing" : "offline"}`} />
            <span className="status-text">
              {backendOk === null ? "Connecting…"
                : backendOk === "online" ? "Backend Online"
                : backendOk === "missing_model" ? "Model Missing"
                : "Backend Offline"}
            </span>
          </div>
          <div className="fps-badge">{fps} FPS</div>
        </div>
      </header>

      {/* ── Main content ── */}
      <main className="app-main">
        {/* Top row: camera (left) + prediction (right) */}
        <div className="top-row">
          <section className="cam-section">
            <WebcamComponent onLandmarks={handleLandmarks} onNoHand={handleNoHand} />
            <p className="cam-hint">Hold a stable sign for 0.5s to add · auto-speaks after pause</p>
          </section>

          <section className="pred-section">
            <div className="panel-label">Prediction</div>
            <Prediction
              result={result}
              noHand={noHand}
              noSign={noSign}
              backendOk={backendOk}
              bufferState={bufferState}
              majorityVote={result?.prediction}
              geoFeatures={geoFeatures}
            />

            <div className="info-grid">
              <div className="info-card">
                <span className="info-icon"></span>
                <span className="info-label">Model</span>
                <span className="info-val">MLP · 93→128→64→29</span>
              </div>
              <div className="info-card">
                <span className="info-icon"></span>
                <span className="info-label">Features</span>
                <span className="info-val">63 landmarks + 30 geo = 93</span>
              </div>
              <div className="info-card">
                <span className="info-icon"></span>
                <span className="info-label">Buffer</span>
                <span className="info-val">Majority vote · last {BUFFER_SIZE}</span>
              </div>
              <div className="info-card">
                <span className="info-icon"></span>
                <span className="info-label">Voice</span>
                <span className="info-val">Auto-speak · 2.5s pause</span>
              </div>
            </div>
          </section>
        </div>

        {/* Word Builder — full width below top row */}
        <div className="word-panel">
          <div className="word-panel-header">
            <span className="word-panel-title">📝 Word Builder</span>
            <div className="word-panel-actions">
              <button
                className={`word-btn speak ${isSpeaking ? "speaking" : ""}`}
                onClick={() => speakWord(word)}
                disabled={!word.trim()}
                title="Speak word"
              >
                {isSpeaking ? "🔊 Speaking…" : "🔊 Speak"}
              </button>
              <button
                className="word-btn backspace"
                onClick={() => { setWord(w => w.slice(0, -1)); playTick(); }}
                disabled={!word.length}
                title="Delete last letter"
              >
                ⌫
              </button>
              <button
                className="word-btn clear"
                onClick={() => { setWord(""); clearTimeout(autoSpeakRef.current); window.speechSynthesis?.cancel(); setIsSpeaking(false); }}
                disabled={!word.length}
                title="Clear all"
              >
                ✕ Clear
              </button>
            </div>
          </div>

          <div className="word-display" ref={displayRef}>
            {word.length > 0
              ? word.split("").map((ch, i) => (
                  <span
                    key={i}
                    className={`word-char ${ch === " " ? "space-char" : ""} ${i === word.length - 1 && lastAdded ? "just-added" : ""}`}
                  >
                    {ch === " " ? "␣" : ch}
                  </span>
                ))
              : <span className="word-placeholder">Sign letters to build a word…</span>
            }
          </div>

          <div className={`hold-bar-wrap ${holdProgress > 0 ? "visible" : ""}`}>
            <div className="hold-bar-track">
              <div className="hold-bar-fill" style={{ width: `${holdProgress}%` }} />
            </div>
            <span className="hold-bar-label">
              {holdProgress > 0 ? `Adding "${holdLetter.current?.toUpperCase()}"…` : ""}
            </span>
          </div>
        </div>
      </main>

      <footer className="app-footer">
        Confusion-Aware Real-Time ASL · MediaPipe Hands · PyTorch MLP · Voice Output
      </footer>
    </div>
  );
}
