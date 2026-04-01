"""
Create a PNG diagram for the project's landmark extraction and preprocessing flow.

Outputs:
  1. ../LandmarkExtractionPipeline.png
  2. ./plots/landmark_extraction_pipeline.png
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
    ROOT / "LandmarkExtractionPipeline.png",
    PLOTS_DIR / "landmark_extraction_pipeline.png",
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
    "rose": "#b45a73",
    "soft_blue": "#e8f1fb",
    "soft_teal": "#e6f4f0",
    "soft_coral": "#f8ece7",
    "soft_gold": "#fbf2de",
    "soft_rose": "#f6eaf0",
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
        ((1.35, 8.7), 1.2, PALETTE["soft_gold"], 0.8),
        ((14.4, 8.4), 1.45, PALETTE["soft_blue"], 0.92),
        ((14.5, 1.25), 1.1, PALETTE["soft_teal"], 0.86),
        ((2.1, 1.1), 1.0, PALETTE["soft_coral"], 0.75),
    ]
    for (x, y), radius, color, alpha in blobs:
        ax.add_patch(
            Circle((x, y), radius, facecolor=color, edgecolor="none", alpha=alpha, zorder=0)
        )


def draw_pill(ax, x, y, text, fill, edge, text_color=None, fontsize=9.0):
    width = 0.22 + 0.088 * len(text)
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
        (x + 0.1, y - 0.1),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.2",
        linewidth=0,
        facecolor=PALETTE["shadow"],
        alpha=0.18,
        zorder=1,
    )
    ax.add_patch(shadow)

    card = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.2",
        linewidth=1.5,
        edgecolor=accent,
        facecolor=PALETTE["white"],
        zorder=2,
    )
    ax.add_patch(card)

    header = FancyBboxPatch(
        (x, y + h - 0.34),
        w,
        0.34,
        boxstyle="round,pad=0.02,rounding_size=0.2",
        linewidth=0,
        facecolor=mix_with_white(accent, 0.87),
        zorder=3,
    )
    ax.add_patch(header)

    badge_circle = Circle((x + 0.34, y + h - 0.17), 0.15, facecolor=accent, edgecolor="none", zorder=4)
    ax.add_patch(badge_circle)
    ax.text(
        x + 0.34,
        y + h - 0.17,
        badge,
        ha="center",
        va="center",
        fontsize=9.6,
        color="white",
        fontweight="bold",
        zorder=5,
    )

    ax.text(
        x + 0.6,
        y + h - 0.17,
        title,
        ha="left",
        va="center",
        fontsize=10.8,
        color=PALETTE["ink"],
        fontweight="bold",
        zorder=5,
    )

    text_y = y + h - 0.57
    for line in lines:
        ax.text(
            x + 0.18,
            text_y,
            line,
            ha="left",
            va="top",
            fontsize=8.45,
            color=PALETTE["muted"],
            zorder=5,
        )
        text_y -= 0.29

    if tag:
        tag_width = 0.3 + 0.082 * len(tag)
        tag_box = FancyBboxPatch(
            (x + w - tag_width - 0.16, y + 0.12),
            tag_width,
            0.28,
            boxstyle="round,pad=0.04,rounding_size=0.14",
            linewidth=0,
            facecolor=mix_with_white(accent, 0.9),
            zorder=4,
        )
        ax.add_patch(tag_box)
        ax.text(
            x + w - tag_width / 2 - 0.16,
            y + 0.26,
            tag,
            ha="center",
            va="center",
            fontsize=7.6,
            color=accent,
            fontweight="bold",
            zorder=5,
        )


def draw_arrow(ax, start, end, color, connection="arc3,rad=0.0", label=None, label_xy=None):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=15,
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
            fontsize=7.7,
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
        "Landmark Extraction and Preprocessing Pipeline",
        ha="center",
        va="center",
        fontsize=20.5,
        color=PALETTE["ink"],
        fontweight="bold",
    )
    ax.text(
        8,
        9.03,
        "How this project turns a live hand frame into a model-ready 93-feature vector",
        ha="center",
        va="center",
        fontsize=11,
        color=PALETTE["muted"],
    )

    draw_pill(ax, 3.1, 8.38, "Frontend capture and landmark tracking", PALETTE["soft_blue"], PALETTE["blue"], PALETTE["blue"])
    draw_pill(ax, 10.2, 5.18, "Backend feature engineering", PALETTE["soft_coral"], PALETTE["coral"], PALETTE["coral"])

    top_y = 6.2
    top_w = 2.55
    top_h = 1.95
    top_x = [0.45, 3.25, 6.05, 8.85, 11.65]

    draw_card(
        ax,
        top_x[0],
        top_y,
        top_w,
        top_h,
        "Webcam frame",
        [
            "React captures the live hand image",
            "and passes frames to MediaPipe Hands.",
        ],
        PALETTE["coral"],
        "1",
        tag="frontend",
    )
    draw_card(
        ax,
        top_x[1],
        top_y,
        top_w,
        top_h,
        "MediaPipe Hands",
        [
            "Single-hand tracking is enabled with",
            "maxNumHands = 1 in the browser.",
        ],
        PALETTE["gold"],
        "2",
        tag="21 keypoints",
    )
    draw_card(
        ax,
        top_x[2],
        top_y,
        top_w,
        top_h,
        "21 landmarks",
        [
            "Each point contains x, y, z values for",
            "the wrist, joints, and fingertips.",
        ],
        PALETTE["blue"],
        "3",
        tag="hand structure",
    )
    draw_card(
        ax,
        top_x[3],
        top_y,
        top_w,
        top_h,
        "Flat 63 vector",
        [
            "The frontend flattens landmarks into",
            "[x0, y0, z0, ... , x20, y20, z20].",
        ],
        PALETTE["teal"],
        "4",
        tag="63 floats",
    )
    draw_card(
        ax,
        top_x[4],
        top_y,
        top_w,
        top_h,
        "Stillness + API",
        [
            "Only stable poses are sent to /predict,",
            "where Flask validates exactly 63 inputs.",
        ],
        PALETTE["slate"],
        "5",
        tag="live inference",
    )

    for idx in range(len(top_x) - 1):
        draw_arrow(
            ax,
            (top_x[idx] + top_w, top_y + 0.95),
            (top_x[idx + 1], top_y + 0.95),
            PALETTE["muted"],
        )

    draw_card(
        ax,
        8.05,
        3.15,
        3.1,
        1.8,
        "Normalize landmarks",
        [
            "Subtract wrist landmark 0 to make",
            "the hand position invariant.",
            "Scale by max absolute magnitude.",
        ],
        PALETTE["coral"],
        "6",
        tag="63 normalized",
    )
    draw_card(
        ax,
        11.45,
        3.15,
        3.1,
        1.8,
        "Geometric features",
        [
            "10 fingertip distances plus 20 joint",
            "angles capture spread and finger bends.",
        ],
        PALETTE["gold"],
        "7",
        tag="30 features",
    )
    draw_card(
        ax,
        9.65,
        0.95,
        3.4,
        1.85,
        "Final feature vector",
        [
            "63 normalized landmarks + 30 geometric",
            "features = 93 values for the MLP model.",
        ],
        PALETTE["teal"],
        "8",
        tag="input to classifier",
    )

    draw_arrow(
        ax,
        (12.925, 6.2),
        (9.6, 4.95),
        PALETTE["coral"],
        connection="arc3,rad=0.08",
        label="backend preprocess()",
        label_xy=(10.1, 5.15),
    )
    draw_arrow(
        ax,
        (12.925, 6.2),
        (12.95, 4.95),
        PALETTE["gold"],
        connection="arc3,rad=-0.02",
    )
    draw_arrow(
        ax,
        (9.6, 3.15),
        (11.35, 2.8),
        PALETTE["coral"],
        connection="arc3,rad=-0.12",
    )
    draw_arrow(
        ax,
        (12.95, 3.15),
        (11.55, 2.8),
        PALETTE["gold"],
        connection="arc3,rad=0.12",
    )

    ax.text(
        8,
        0.2,
        "The same landmark style is used in training and live inference, so the model sees a consistent hand representation.",
        ha="center",
        va="bottom",
        fontsize=9.35,
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
