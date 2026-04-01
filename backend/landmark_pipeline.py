"""
landmark_pipeline.py
Draws the full Landmark Extraction Pipeline diagram and saves as PNG.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

fig, ax = plt.subplots(figsize=(18, 11))
ax.set_xlim(0, 18)
ax.set_ylim(0, 11)
ax.axis("off")
fig.patch.set_facecolor("#0d1117")
ax.set_facecolor("#0d1117")

# ── Colors ────────────────────────────────────────────────────────────────────
BG    = "#0d1117"
C = {
    "webcam":   "#1f77b4",
    "mp":       "#17becf",
    "raw":      "#2ca02c",
    "norm":     "#9467bd",
    "geo":      "#ff7f0e",
    "concat":   "#e377c2",
    "mlp":      "#d62728",
    "output":   "#bcbd22",
    "arrow":    "#555566",
    "text":     "white",
    "dim":      "#888899",
    "title":    "#00d4ff",
}

# ── Helper: rounded box ───────────────────────────────────────────────────────
def box(ax, x, y, w, h, label, sublabel, color, fontsize=10):
    rect = mpatches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.12",
        linewidth=2, edgecolor=color,
        facecolor=color + "22",
        zorder=3
    )
    ax.add_patch(rect)
    ty = y + h / 2 + (0.12 if sublabel else 0)
    ax.text(x + w/2, ty, label,
            ha="center", va="center", fontsize=fontsize,
            fontweight="bold", color=color, zorder=4)
    if sublabel:
        ax.text(x + w/2, y + h/2 - 0.22, sublabel,
                ha="center", va="center", fontsize=7.5,
                color=C["dim"], zorder=4)

def arrow_h(ax, x1, x2, y, color=C["arrow"]):
    ax.annotate("", xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=1.8, mutation_scale=14),
                zorder=5)

def arrow_v(ax, x, y1, y2, color=C["arrow"]):
    ax.annotate("", xy=(x, y2), xytext=(x, y1),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=1.8, mutation_scale=14),
                zorder=5)

def label_arrow(ax, x, y, text, color=C["dim"]):
    ax.text(x, y, text, ha="center", va="center",
            fontsize=7, color=color, zorder=6,
            bbox=dict(boxstyle="round,pad=0.2", fc=BG, ec="none"))

# ═══════════════════════════════════════════════════════════════════════════════
# ROW 1 — Input pipeline  (y=8.2)
# ═══════════════════════════════════════════════════════════════════════════════
ROW1 = 7.8

# 1. Webcam / Image Input
box(ax, 0.3, ROW1, 2.4, 1.4, "Webcam / Image", "640×480 RGB frame", C["webcam"])

arrow_h(ax, 2.7, 3.2, ROW1 + 0.7)
label_arrow(ax, 2.95, ROW1 + 0.95, "RGB")

# 2. MediaPipe Hands
box(ax, 3.2, ROW1, 3.0, 1.4, "MediaPipe Hands", "Hand Landmarker · Tasks API", C["mp"])

arrow_h(ax, 6.2, 6.7, ROW1 + 0.7)
label_arrow(ax, 6.45, ROW1 + 0.95, "21 pts")

# 3. Raw Landmarks
box(ax, 6.7, ROW1, 2.8, 1.4, "Raw Landmarks", "21 × (x, y, z)  →  63 floats", C["raw"])

# ── MediaPipe detail box ──────────────────────────────────────────────────────
mp_detail_x, mp_detail_y = 3.2, 5.9
rect2 = mpatches.FancyBboxPatch(
    (mp_detail_x, mp_detail_y), 3.0, 1.6,
    boxstyle="round,pad=0.1", linewidth=1,
    edgecolor=C["mp"] + "88", facecolor=C["mp"] + "11", zorder=2
)
ax.add_patch(rect2)
ax.text(mp_detail_x + 1.5, mp_detail_y + 1.25, "MediaPipe Config",
        ha="center", fontsize=7.5, fontweight="bold", color=C["mp"])
for i, line in enumerate([
    "maxNumHands = 1",
    "modelComplexity = 1",
    "minDetectionConf = 0.6",
    "minTrackingConf  = 0.5",
]):
    ax.text(mp_detail_x + 0.2, mp_detail_y + 0.95 - i*0.22, "• " + line,
            fontsize=6.8, color=C["dim"])
arrow_v(ax, mp_detail_x + 1.5, mp_detail_y + 1.6, ROW1)

# ═══════════════════════════════════════════════════════════════════════════════
# ROW 2 — Preprocessing split  (y=4.0)
# ═══════════════════════════════════════════════════════════════════════════════
ROW2 = 3.8

# Arrow down from Raw Landmarks
arrow_v(ax, 8.1, ROW1, ROW2 + 1.4)
label_arrow(ax, 8.1, (ROW1 + ROW2 + 1.4)/2, "63 floats")

# Split line
ax.plot([5.5, 10.7], [ROW2 + 1.4, ROW2 + 1.4],
        color=C["dim"], lw=1.2, linestyle="--", zorder=2)

# Branch A — Normalization
arrow_v(ax, 6.2, ROW2 + 1.4, ROW2 + 1.0)
box(ax, 4.8, ROW2 - 0.2, 2.8, 1.4,
    "Normalize", "Subtract wrist · Scale ÷ max", C["norm"])

# Normalization detail
norm_dx, norm_dy = 4.8, 1.8
rect3 = mpatches.FancyBboxPatch(
    (norm_dx, norm_dy), 2.8, 1.7,
    boxstyle="round,pad=0.1", linewidth=1,
    edgecolor=C["norm"] + "88", facecolor=C["norm"] + "11", zorder=2
)
ax.add_patch(rect3)
ax.text(norm_dx + 1.4, norm_dy + 1.42, "Normalization Steps",
        ha="center", fontsize=7.5, fontweight="bold", color=C["norm"])
for i, line in enumerate([
    "1. Reshape → (21, 3)",
    "2. lm -= wrist[0]  (origin shift)",
    "3. lm /= max(|lm|)  → [-1, 1]",
    "4. Flatten → 63 floats",
]):
    ax.text(norm_dx + 0.15, norm_dy + 1.15 - i*0.25, "• " + line,
            fontsize=6.5, color=C["dim"])
arrow_v(ax, norm_dx + 1.4, norm_dy + 1.7, ROW2 - 0.2)

# Branch B — Geometric Features
arrow_v(ax, 10.0, ROW2 + 1.4, ROW2 + 1.0)
box(ax, 8.6, ROW2 - 0.2, 2.8, 1.4,
    "Geometric Features", "Fingertip distances + Joint angles", C["geo"])

# Geometric detail
geo_dx, geo_dy = 8.6, 1.8
rect4 = mpatches.FancyBboxPatch(
    (geo_dx, geo_dy), 2.8, 1.7,
    boxstyle="round,pad=0.1", linewidth=1,
    edgecolor=C["geo"] + "88", facecolor=C["geo"] + "11", zorder=2
)
ax.add_patch(rect4)
ax.text(geo_dx + 1.4, geo_dy + 1.42, "Geometric Features",
        ha="center", fontsize=7.5, fontweight="bold", color=C["geo"])
for i, line in enumerate([
    "Fingertips: {4, 8, 12, 16, 20}",
    "10 pairwise distances (C(5,2))",
    "5 fingers × 4 joints = 20 angles",
    "Total → 30 floats",
]):
    ax.text(geo_dx + 0.15, geo_dy + 1.15 - i*0.25, "• " + line,
            fontsize=6.5, color=C["dim"])
arrow_v(ax, geo_dx + 1.4, geo_dy + 1.7, ROW2 - 0.2)

# ═══════════════════════════════════════════════════════════════════════════════
# ROW 3 — Concatenation + MLP + Output  (y=1.0)
# ═══════════════════════════════════════════════════════════════════════════════
ROW3 = 0.5

# Arrows down from both branches to concat
arrow_v(ax, 6.2,  ROW2 - 0.2, ROW3 + 1.4)
arrow_v(ax, 10.0, ROW2 - 0.2, ROW3 + 1.4)

# Concat merge line
ax.plot([6.2, 10.0], [ROW3 + 1.4, ROW3 + 1.4],
        color=C["dim"], lw=1.2, linestyle="--", zorder=2)

# Concat box
box(ax, 7.0, ROW3, 2.0, 1.2, "Concatenate", "63 + 30 = 93 floats", C["concat"])
arrow_v(ax, 8.0, ROW3 + 1.4, ROW3 + 1.2)

# Arrow to MLP
arrow_h(ax, 9.0, 9.6, ROW3 + 0.6)
label_arrow(ax, 9.3, ROW3 + 0.9, "93-d")

# MLP box
box(ax, 9.6, ROW3, 2.4, 1.2, "MLP Classifier", "Linear→BN→ReLU×3", C["mlp"])

# Arrow to Output
arrow_h(ax, 12.0, 12.6, ROW3 + 0.6)
label_arrow(ax, 12.3, ROW3 + 0.9, "logits")

# Output box
box(ax, 12.6, ROW3, 3.0, 1.2, "Prediction", "Softmax → 29 classes\nA–Z + del + space + nothing", C["output"])

# ── Stillness / Buffer note ───────────────────────────────────────────────────
buf_x, buf_y = 13.5, 4.5
rect5 = mpatches.FancyBboxPatch(
    (buf_x, buf_y), 3.8, 2.8,
    boxstyle="round,pad=0.12", linewidth=1.5,
    edgecolor="#444455", facecolor="#1a1a2e", zorder=2
)
ax.add_patch(rect5)
ax.text(buf_x + 1.9, buf_y + 2.5, "Runtime Stability",
        ha="center", fontsize=9, fontweight="bold", color="#00d4ff")
for i, line in enumerate([
    "Frame skip: every 2nd frame",
    "Stillness threshold: 0.025",
    "Still frames required: 3",
    "Sliding buffer: last 5 preds",
    "Final output: majority vote",
]):
    ax.text(buf_x + 0.2, buf_y + 2.1 - i*0.38, "▸ " + line,
            fontsize=7.5, color=C["dim"])

# ── Hand skeleton diagram ─────────────────────────────────────────────────────
sk_cx, sk_cy = 15.5, 8.2
# Approximate 21 landmark positions (normalized hand shape)
lm_pos = {
    0:  (0.0,  0.0),
    1:  (-0.3, 0.5), 2: (-0.35, 0.9), 3: (-0.38, 1.2), 4: (-0.4, 1.5),
    5:  (-0.1, 1.1), 6: (-0.1, 1.5),  7: (-0.1, 1.8),  8: (-0.1, 2.1),
    9:  (0.1,  1.1), 10:(0.1,  1.55), 11:(0.1,  1.85), 12:(0.1,  2.15),
    13: (0.3,  1.0), 14:(0.3,  1.45), 15:(0.3,  1.75), 16:(0.3,  2.0),
    17: (0.5,  0.8), 18:(0.5,  1.2),  19:(0.5,  1.5),  20:(0.5,  1.75),
}
scale_sk = 0.55
connections = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]
fingertips = {4, 8, 12, 16, 20}
for a, b in connections:
    x1 = sk_cx + lm_pos[a][0] * scale_sk
    y1 = sk_cy + lm_pos[a][1] * scale_sk
    x2 = sk_cx + lm_pos[b][0] * scale_sk
    y2 = sk_cy + lm_pos[b][1] * scale_sk
    ax.plot([x1, x2], [y1, y2], color="#336699", lw=1.2, zorder=3)
for idx, (px, py) in lm_pos.items():
    x = sk_cx + px * scale_sk
    y = sk_cy + py * scale_sk
    color = C["geo"] if idx in fingertips else (C["mp"] if idx == 0 else "#aaaacc")
    size  = 40 if idx in fingertips else (55 if idx == 0 else 18)
    ax.scatter(x, y, s=size, color=color, zorder=4)
    ax.text(x + 0.06, y + 0.04, str(idx), fontsize=5, color="white", zorder=5)

ax.text(sk_cx, sk_cy - 0.25, "21 Landmarks (MediaPipe)",
        ha="center", fontsize=7.5, color=C["mp"])
ax.text(sk_cx, sk_cy - 0.52, "● Wrist  ● Fingertips  · Joints",
        ha="center", fontsize=6.5, color=C["dim"])

# ── Title ─────────────────────────────────────────────────────────────────────
ax.text(9, 10.65, "Landmark Extraction Pipeline — ASL Gesture Recognition",
        ha="center", fontsize=15, fontweight="bold", color=C["title"])
ax.text(9, 10.25, "Webcam → MediaPipe → Normalize + Geometric Features → Concatenate → MLP → Prediction",
        ha="center", fontsize=8.5, color=C["dim"])

# ── Step numbers ──────────────────────────────────────────────────────────────
for i, (sx, sy, label) in enumerate([
    (1.5,  9.55, "①"),
    (4.7,  9.55, "②"),
    (8.1,  9.55, "③"),
    (6.2,  4.95, "④a"),
    (10.0, 4.95, "④b"),
    (8.0,  1.95, "⑤"),
    (10.8, 1.95, "⑥"),
    (14.1, 1.95, "⑦"),
]):
    ax.text(sx, sy, label, ha="center", va="center",
            fontsize=8, color="#00d4ff",
            bbox=dict(boxstyle="circle,pad=0.15", fc="#00d4ff22", ec="#00d4ff55"))

# ── Save ──────────────────────────────────────────────────────────────────────
out = os.path.join(os.path.dirname(__file__), "plots", "landmark_pipeline.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
plt.savefig(out, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"Saved: {out}")
