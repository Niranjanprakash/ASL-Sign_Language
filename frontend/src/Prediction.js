import "./Prediction.css";
import GeometryPanel from "./GeometryPanel";

const STATUS_MAP = {
  stable: { icon: "✓", label: "STABLE" },
  uncertain: { icon: "⚠", label: "UNCERTAIN" },
  confused: { icon: "↔", label: "CONFUSED" },
};

export default function Prediction({ result, noHand, noSign, backendOk, bufferState = [], majorityVote, geoFeatures }) {
  if (backendOk === null) {
    return (
      <div className="prediction-card no-hand">
        <div className="no-hand-icon">⏳</div>
        <p className="no-hand-text">Connecting to Backend…</p>
        <p className="no-hand-sub">Render free tier may take up to 60s on first load</p>
      </div>
    );
  }
  if (backendOk === "missing_model") {
    return (
      <div className="prediction-card no-hand">
        <div className="no-hand-icon">⚠️</div>
        <p className="no-hand-text">No Model Found</p>
        <p className="no-hand-sub">Run train.py to build the network</p>
      </div>
    );
  }

  if (backendOk === "offline") {
    return (
      <div className="prediction-card no-hand">
        <div className="no-hand-icon">🔌</div>
        <p className="no-hand-text">Backend Offline</p>
        <p className="no-hand-sub">Start app.py to connect</p>
      </div>
    );
  }

  if (noHand) {
    return (
      <div className="prediction-card no-hand">
        <div className="no-hand-icon">🤚</div>
        <p className="no-hand-text">No Hand Detected</p>
        <p className="no-hand-sub">Position hand clearly in frame</p>
      </div>
    );
  }

  if (noSign) {
    return (
      <div className="prediction-card no-hand">
        <div className="no-hand-icon">✋</div>
        <p className="no-hand-text">No Sign Detected</p>
        <p className="no-hand-sub">Hold your hand still to sign a letter</p>
      </div>
    );
  }

  if (!result) {
    return (
      <div className="prediction-card no-hand">
        <div className="no-hand-icon">...</div>
        <p className="no-hand-text">Initializing</p>
        <p className="no-hand-sub">Warming up model</p>
      </div>
    );
  }

  const { prediction, confidence, status, possible_confusions = [], adaptive } = result;
  const confPct = Math.round(confidence * 100);
  const statusInfo = STATUS_MAP[status] || { icon: "?", label: "UNKNOWN" };
  const isLongText = prediction.length > 3;

  return (
    <div className="prediction-card">
      <div className="pred-header">
        <div>
          <div className="pred-main-label">Detected Output</div>
          <div className="pred-letter" key={prediction} style={{ fontSize: isLongText ? "3.2rem" : undefined }}>
            {prediction.toUpperCase()}
          </div>
        </div>
        <div className={`pred-status ${status}`}>
          <span style={{ fontSize: "0.9rem" }}>{statusInfo.icon}</span>
          {statusInfo.label}
        </div>
      </div>

      <div className="conf-section">
        <div className="conf-label">
          <span>Confidence</span>
          <span className="conf-val">{confPct}%</span>
        </div>
        <div className="conf-track">
          <div className="conf-fill" style={{ width: `${confPct}%` }} />
        </div>
      </div>

      {/* Adaptive threshold badge */}
      {adaptive && (
        <div className="adaptive-badge">
          <span>🔍</span> High-Scrutiny Letter — Stricter confidence threshold applied
        </div>
      )}

      {/* Confusion warning */}
      {status === "confused" && possible_confusions.length > 0 && (
        <div className="confusion-alert">
          <div className="confuse-icon">👀</div>
          <div className="confuse-text">
            Unclear posture. Could also be: <b>{possible_confusions.join(", ")}</b>
          </div>
        </div>
      )}

      {/* Majority vote buffer */}
      {bufferState.length > 0 && (
        <div className="buffer-row">
          <span className="buffer-label">Buffer:</span>
          {bufferState.map((v, i) => (
            <span key={i} className={`buffer-cell ${v === majorityVote ? "match" : ""}`}>{v}</span>
          ))}
          <span className="buffer-arrow">→</span>
          <span className="buffer-result">{majorityVote}</span>
        </div>
      )}

      {/* Geometry Panel */}
      <GeometryPanel geometric={geoFeatures} />
    </div>
  );
}
