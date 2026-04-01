import "./GeometryPanel.css";

const FINGERTIP_PAIRS = [
  "T↔I", "T↔M", "T↔R", "T↔P",
  "I↔M", "I↔R", "I↔P",
  "M↔R", "M↔P",
  "R↔P",
];

const FINGER_NAMES = ["Thumb", "Index", "Middle", "Ring", "Pinky"];
const ANGLE_JOINT_LABELS = ["Wrist-MCP", "MCP-PIP", "PIP-DIP"];

function normalize(val, max) {
  return Math.min((val / max) * 100, 100);
}

function bendLabel(angleRad) {
  const deg = (angleRad * 180) / Math.PI;
  if (deg > 150) return "Straight";
  if (deg > 110) return "Slight";
  if (deg > 70)  return "Curved";
  return "Bent";
}

export default function GeometryPanel({ geometric }) {
  if (!geometric) return null;
  const { distances = [], angles = [] } = geometric;

  // max distance for normalizing bars
  const maxDist = Math.max(...distances, 0.01);

  // group angles: 3 per finger × 5 fingers = 15
  const fingerAngles = FINGER_NAMES.map((_, fi) =>
    angles.slice(fi * 3, fi * 3 + 3)
  );

  return (
    <div className="geo-panel">
      <div className="geo-title">🖐 Hand Geometry</div>

      {/* ── Fingertip Distances ── */}
      <div className="geo-section-label">Fingertip Distances</div>
      <div className="geo-distances">
        {distances.map((d, i) => (
          <div key={i} className="geo-dist-row">
            <span className="geo-pair-label">{FINGERTIP_PAIRS[i]}</span>
            <div className="geo-bar-track">
              <div
                className="geo-bar-fill dist-fill"
                style={{ width: `${normalize(d, maxDist)}%` }}
              />
            </div>
            <span className="geo-val">{d.toFixed(2)}</span>
          </div>
        ))}
      </div>

      {/* ── Finger Bend Angles ── */}
      <div className="geo-section-label">Finger Bend</div>
      <div className="geo-angles">
        {fingerAngles.map((fa, fi) => {
          const avgAngle = fa.length ? fa.reduce((s, v) => s + v, 0) / fa.length : 0;
          const pct = normalize(avgAngle, Math.PI);
          return (
            <div key={fi} className="geo-angle-row">
              <span className="geo-finger-label">{FINGER_NAMES[fi]}</span>
              <div className="geo-bar-track">
                <div
                  className="geo-bar-fill angle-fill"
                  style={{ width: `${pct}%` }}
                />
              </div>
              <span className={`geo-bend-label bend-${bendLabel(avgAngle).toLowerCase()}`}>
                {bendLabel(avgAngle)}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
