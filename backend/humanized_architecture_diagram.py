"""
Create a human-centered system architecture figure for the ASL project.

The figure is exported to:
  1. ../Architecture.png
  2. ./plots/humanized_architecture.png
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
PLOTS_DIR = Path(__file__).resolve().parent / "plots"
OUTPUTS = [
    ROOT / "Architecture.png",
    PLOTS_DIR / "humanized_architecture.png",
]


PALETTE = {
    "bg": "#f7f3ec",
    "ink": "#24323d",
    "muted": "#5f6c76",
    "shadow": "#c9c1b6",
    "blue": "#4f7cac",
    "teal": "#2f7d6b",
    "coral": "#d96c4f",
    "gold": "#d8a23d",
    "slate": "#6a7f91",
    "soft_blue": "#e8f1fb",
    "soft_teal": "#e6f4f0",
    "soft_coral": "#f8ece7",
    "soft_gold": "#fbf2de",
    "white": "#ffffff",
}


def mix_with_white(color: str, white_weight: float = 0.9):
    r, g, b = to_rgb(color)
    return (
        white_weight + (1 - white_weight) * r,
        white_weight + (1 - white_weight) * g,
        white_weight + (1 - white_weight) * b,
    )


def add_background(ax):
    blobs = [
        ((1.6, 8.5), 1.3, PALETTE["soft_gold"], 0.8),
        ((14.2, 8.1), 1.5, PALETTE["soft_blue"], 0.95),
        ((14.7, 1.6), 1.1, PALETTE["soft_teal"], 0.9),
        ((2.4, 1.5), 1.1, PALETTE["soft_coral"], 0.8),
    ]
    for (x, y), radius, color, alpha in blobs:
        ax.add_patch(
            Circle((x, y), radius, facecolor=color, edgecolor="none", alpha=alpha, zorder=0)
        )


def draw_pill(ax, x, y, text, fill, edge, text_color=None, fontsize=9.5):
    width = 0.2 + 0.09 * len(text)
    patch = FancyBboxPatch(
        (x, y),
        width,
        0.38,
        boxstyle="round,pad=0.04,rounding_size=0.18",
        linewidth=1.0,
        edgecolor=edge,
        facecolor=fill,
        zorder=5,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2,
        y + 0.19,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=text_color or PALETTE["ink"],
        zorder=6,
        fontweight="bold",
    )


def draw_card(ax, x, y, w, h, title, lines, accent, badge, tag=None):
    shadow = FancyBboxPatch(
        (x + 0.12, y - 0.12),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.22",
        linewidth=0,
        facecolor=PALETTE["shadow"],
        alpha=0.22,
        zorder=1,
    )
    ax.add_patch(shadow)

    card = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.22",
        linewidth=1.6,
        edgecolor=accent,
        facecolor=PALETTE["white"],
        zorder=2,
    )
    ax.add_patch(card)

    header = FancyBboxPatch(
        (x, y + h - 0.34),
        w,
        0.34,
        boxstyle="round,pad=0.02,rounding_size=0.22",
        linewidth=0,
        facecolor=mix_with_white(accent, 0.86),
        zorder=3,
    )
    ax.add_patch(header)

    badge_circle = Circle((x + 0.36, y + h - 0.17), 0.16, facecolor=accent, edgecolor="none", zorder=4)
    ax.add_patch(badge_circle)
    ax.text(
        x + 0.36,
        y + h - 0.17,
        badge,
        ha="center",
        va="center",
        fontsize=10,
        color="white",
        fontweight="bold",
        zorder=5,
    )

    ax.text(
        x + 0.65,
        y + h - 0.18,
        title,
        ha="left",
        va="center",
        fontsize=12.5,
        color=PALETTE["ink"],
        fontweight="bold",
        zorder=5,
    )

    text_y = y + h - 0.58
    for line in lines:
        ax.text(
            x + 0.24,
            text_y,
            line,
            ha="left",
            va="top",
            fontsize=9.4,
            color=PALETTE["muted"],
            zorder=5,
        )
        text_y -= 0.34

    if tag:
        tag_width = 0.3 + 0.085 * len(tag)
        tag_box = FancyBboxPatch(
            (x + w - tag_width - 0.18, y + 0.14),
            tag_width,
            0.3,
            boxstyle="round,pad=0.04,rounding_size=0.14",
            linewidth=0,
            facecolor=mix_with_white(accent, 0.9),
            zorder=4,
        )
        ax.add_patch(tag_box)
        ax.text(
            x + w - tag_width / 2 - 0.18,
            y + 0.29,
            tag,
            ha="center",
            va="center",
            fontsize=8.2,
            color=accent,
            fontweight="bold",
            zorder=5,
        )


def draw_arrow(ax, start, end, color, connection="arc3,rad=0.0", style="-|>", label=None, label_xy=None):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=16,
        linewidth=1.7,
        color=color,
        connectionstyle=connection,
        zorder=3,
    )
    ax.add_patch(arrow)
    if label and label_xy:
        draw_pill(
            ax,
            label_xy[0],
            label_xy[1],
            label,
            fill=mix_with_white(color, 0.9),
            edge=color,
            text_color=color,
            fontsize=8.3,
        )


def main():
    plt.rcParams["font.family"] = "DejaVu Sans"

    fig, ax = plt.subplots(figsize=(16, 10), dpi=220)
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis("off")

    add_background(ax)

    ax.text(
        8,
        9.45,
        "Human-Centered ASL Recognition Workflow",
        ha="center",
        va="center",
        fontsize=21,
        color=PALETTE["ink"],
        fontweight="bold",
    )
    ax.text(
        8,
        9.05,
        "From a person's hand sign to stable letters, word building, and speech output",
        ha="center",
        va="center",
        fontsize=11.2,
        color=PALETTE["muted"],
    )

    draw_pill(ax, 6.82, 8.4, "Live application flow", PALETTE["soft_blue"], PALETTE["blue"], PALETTE["blue"])
    draw_pill(ax, 6.95, 1.92, "Offline training", PALETTE["soft_coral"], PALETTE["coral"], PALETTE["coral"])

    draw_card(
        ax,
        0.8,
        6.45,
        3.15,
        1.35,
        "Person signs a letter",
        [
            "User presents one hand to the camera.",
            "The goal is a natural ASL gesture, not a perfect pose.",
        ],
        PALETTE["coral"],
        "1",
    )
    draw_card(
        ax,
        0.8,
        4.62,
        3.15,
        1.35,
        "Live webcam view",
        [
            "React mirrors the camera feed for a natural experience.",
            "An overlay helps the user stay centered in frame.",
        ],
        PALETTE["gold"],
        "2",
    )
    draw_card(
        ax,
        0.8,
        2.79,
        3.15,
        1.35,
        "MediaPipe hand landmarks",
        [
            "21 hand keypoints are tracked from each frame.",
            "The browser flattens them into a landmark vector.",
        ],
        PALETTE["teal"],
        "3",
    )

    draw_card(
        ax,
        4.65,
        5.18,
        6.15,
        2.45,
        "Browser interaction layer",
        [
            "Checks stillness so moving hands are not committed too early.",
            "Draws a hand skeleton overlay for instant visual feedback.",
            "Buffers recent predictions and keeps the majority vote.",
        ],
        PALETTE["blue"],
        "4",
        tag="React + MediaPipe JS",
    )
    draw_card(
        ax,
        4.65,
        2.3,
        6.15,
        2.45,
        "Recognition engine",
        [
            "Flask receives landmarks and prepares a feature vector.",
            "Landmarks are normalized and enriched with hand geometry.",
            "A PyTorch MLP predicts the label, confidence, and confusions.",
        ],
        PALETTE["teal"],
        "5",
        tag="Flask API + PyTorch",
    )

    draw_card(
        ax,
        11.45,
        6.62,
        3.7,
        1.1,
        "Stable letter result",
        [
            "Returns A-Z plus special commands like del and space.",
        ],
        PALETTE["slate"],
        "6",
    )
    draw_card(
        ax,
        11.45,
        5.05,
        3.7,
        1.1,
        "Confidence hints",
        [
            "Warns when similar signs such as A, S, and E overlap.",
        ],
        PALETTE["coral"],
        "7",
    )
    draw_card(
        ax,
        11.45,
        3.48,
        3.7,
        1.1,
        "Word builder",
        [
            "Hold-to-commit logic turns letters into a growing word.",
        ],
        PALETTE["gold"],
        "8",
    )
    draw_card(
        ax,
        11.45,
        1.91,
        3.7,
        1.1,
        "Speech output",
        [
            "After a pause, the app can read the signed word aloud.",
        ],
        PALETTE["teal"],
        "9",
    )

    draw_card(
        ax,
        4.95,
        0.45,
        5.55,
        1.18,
        "Offline training pipeline",
        [
            "ASL image dataset -> landmark extraction -> trained MLP checkpoint.",
            "The saved model is loaded by the backend for real-time inference.",
        ],
        PALETTE["coral"],
        "T",
        tag="Kaggle ASL dataset",
    )

    draw_arrow(ax, (3.95, 7.12), (4.65, 6.93), PALETTE["coral"])
    draw_arrow(ax, (3.95, 5.29), (4.65, 6.37), PALETTE["gold"])
    draw_arrow(ax, (3.95, 3.46), (4.65, 5.78), PALETTE["teal"])
    draw_arrow(
        ax,
        (7.72, 5.18),
        (7.72, 4.75),
        PALETTE["blue"],
        label="63 landmarks sent to /predict",
        label_xy=(6.12, 4.86),
    )
    draw_arrow(
        ax,
        (7.72, 1.63),
        (7.72, 2.3),
        PALETTE["coral"],
        label="trained checkpoint loaded by backend",
        label_xy=(5.95, 1.7),
    )

    output_targets = [7.17, 5.6, 4.03, 2.46]
    for y_target, color in zip(
        output_targets,
        [PALETTE["slate"], PALETTE["coral"], PALETTE["gold"], PALETTE["teal"]],
    ):
        draw_arrow(ax, (10.8, 3.52), (11.45, y_target), color)

    ax.text(
        8,
        0.12,
        "The pipeline is designed to feel supportive to the signer: stable recognition first, then text, then voice.",
        ha="center",
        va="bottom",
        fontsize=9.5,
        color=PALETTE["muted"],
    )

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    for output_path in OUTPUTS:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight", pad_inches=0.18, facecolor=fig.get_facecolor())

    plt.close(fig)
    print("Saved outputs:")
    for output_path in OUTPUTS:
        print(f"- {output_path}")


if __name__ == "__main__":
    main()
