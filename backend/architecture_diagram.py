"""
architecture_diagram.py
Draws the ASL MLP architecture diagram and saves as PNG.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

fig, ax = plt.subplots(figsize=(10, 16))
ax.set_xlim(0, 10)
ax.set_ylim(0, 20)
ax.axis("off")
fig.patch.set_facecolor("#0f1117")
ax.set_facecolor("#0f1117")

# ── Color palette ─────────────────────────────────────────────────────────────
C = {
    "input":    "#1f77b4",
    "linear":   "#2ca02c",
    "bn":       "#9467bd",
    "relu":     "#ff7f0e",
    "dropout":  "#d62728",
    "output":   "#e377c2",
    "arrow":    "#aaaaaa",
    "text":     "white",
    "title":    "#00d4ff",
    "sub":      "#888888",
}

def box(ax, y, label, sublabel, color, height=0.7):
    rect = mpatches.FancyBboxPatch(
        (1.5, y), 7, height,
        boxstyle="round,pad=0.08",
        linewidth=1.5,
        edgecolor=color,
        facecolor=color + "33",  # transparent fill
    )
    ax.add_patch(rect)
    ax.text(5, y + height / 2 + 0.05, label,
            ha="center", va="center", fontsize=12,
            fontweight="bold", color=color)
    if sublabel:
        ax.text(5, y + height / 2 - 0.22, sublabel,
                ha="center", va="center", fontsize=8.5, color=C["sub"])

def arrow(ax, y_from, y_to):
    ax.annotate("", xy=(5, y_to), xytext=(5, y_from),
                arrowprops=dict(arrowstyle="-|>", color=C["arrow"],
                                lw=1.5, mutation_scale=14))

# ── Layer definitions: (y_bottom, label, sublabel, color) ────────────────────
layers = [
    (17.8, "INPUT",          "93 features  ·  63 normalized landmarks + 30 geometric", C["input"]),
    (15.8, "Linear",         "93  →  256",                                              C["linear"]),
    (14.6, "BatchNorm1d",    "256",                                                     C["bn"]),
    (13.4, "ReLU",           "",                                                        C["relu"]),
    (12.2, "Dropout",        "p = 0.3",                                                 C["dropout"]),
    (10.2, "Linear",         "256  →  128",                                             C["linear"]),
    ( 9.0, "BatchNorm1d",    "128",                                                     C["bn"]),
    ( 7.8, "ReLU",           "",                                                        C["relu"]),
    ( 6.6, "Dropout",        "p = 0.2",                                                 C["dropout"]),
    ( 4.6, "Linear",         "128  →  64",                                              C["linear"]),
    ( 3.4, "BatchNorm1d",    "64",                                                      C["bn"]),
    ( 2.2, "ReLU",           "",                                                        C["relu"]),
    ( 0.4, "OUTPUT  Linear", "64  →  29   ·   A–Z + del + space + nothing",            C["output"]),
]

# Draw boxes and arrows
for i, (y, label, sub, color) in enumerate(layers):
    box(ax, y, label, sub, color)
    if i < len(layers) - 1:
        arrow(ax, y, layers[i + 1][0] + 0.7)

# ── Block brackets ────────────────────────────────────────────────────────────
def bracket(ax, y_top, y_bot, label, color):
    x = 9.0
    ax.plot([x, x+0.15, x+0.15, x], [y_top, y_top, y_bot, y_bot],
            color=color, lw=1.2)
    ax.text(x + 0.35, (y_top + y_bot) / 2, label,
            va="center", ha="left", fontsize=7.5, color=color, rotation=90)

bracket(ax, 18.5, 12.2, "Block 1", C["linear"])
bracket(ax, 10.9, 6.6,  "Block 2", C["linear"])
bracket(ax, 5.3,  2.2,  "Block 3", C["linear"])

# ── Title ─────────────────────────────────────────────────────────────────────
ax.text(5, 19.5, "ASL MLP — Model Architecture",
        ha="center", va="center", fontsize=15,
        fontweight="bold", color=C["title"])
ax.text(5, 19.0, "Lightweight MLP · 29-Class ASL Alphabet Recognition",
        ha="center", va="center", fontsize=9, color=C["sub"])

# ── Legend ────────────────────────────────────────────────────────────────────
legend_items = [
    ("Linear Layer",  C["linear"]),
    ("BatchNorm1d",   C["bn"]),
    ("ReLU",          C["relu"]),
    ("Dropout",       C["dropout"]),
    ("Input / Output",C["input"]),
]
for i, (lbl, col) in enumerate(legend_items):
    ax.add_patch(mpatches.Rectangle((0.3 + i*1.9, 0.05), 0.25, 0.22,
                                     facecolor=col+"55", edgecolor=col, lw=1))
    ax.text(0.62 + i*1.9, 0.16, lbl, fontsize=6.5, color=col, va="center")

import os
out = os.path.join(os.path.dirname(__file__), "plots", "architecture_diagram.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
plt.savefig(out, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"Saved: {out}")
