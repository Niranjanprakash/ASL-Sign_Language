"""
Create a human-centered figure for the existing vision-based sign recognition workflow.

Outputs:
  1. ../ExistingVisionWorkflow.png
  2. ./plots/existing_vision_workflow_humanized.png
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
    ROOT / "ExistingVisionWorkflow.png",
    PLOTS_DIR / "existing_vision_workflow_humanized.png",
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
    "warning": "#ba5a44",
    "soft_blue": "#e8f1fb",
    "soft_teal": "#e6f4f0",
    "soft_coral": "#f8ece7",
    "soft_gold": "#fbf2de",
    "soft_rose": "#f6eaf0",
    "soft_warning": "#faece7",
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
        ((1.5, 8.8), 1.2, PALETTE["soft_gold"], 0.85),
        ((14.4, 8.4), 1.45, PALETTE["soft_blue"], 0.92),
        ((14.1, 1.35), 1.15, PALETTE["soft_teal"], 0.88),
        ((2.1, 1.45), 1.1, PALETTE["soft_coral"], 0.8),
    ]
    for (x, y), radius, color, alpha in blobs:
        ax.add_patch(
            Circle((x, y), radius, facecolor=color, edgecolor="none", alpha=alpha, zorder=0)
        )


def draw_pill(ax, x, y, text, fill, edge, text_color=None, fontsize=9.2):
    width = 0.2 + 0.088 * len(text)
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
        (x + 0.1, y - 0.11),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.2",
        linewidth=0,
        facecolor=PALETTE["shadow"],
        alpha=0.2,
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
        fontsize=11.2,
        color=PALETTE["ink"],
        fontweight="bold",
        zorder=5,
    )

    text_y = y + h - 0.58
    for line in lines:
        ax.text(
            x + 0.2,
            text_y,
            line,
            ha="left",
            va="top",
            fontsize=8.8,
            color=PALETTE["muted"],
            zorder=5,
        )
        text_y -= 0.31

    if tag:
        tag_width = 0.3 + 0.084 * len(tag)
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
            fontsize=7.8,
            color=accent,
            fontweight="bold",
            zorder=5,
        )


def draw_limitation(ax, x, y, w, h, title, lines, accent, badge):
    shadow = FancyBboxPatch(
        (x + 0.08, y - 0.09),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.18",
        linewidth=0,
        facecolor=PALETTE["shadow"],
        alpha=0.16,
        zorder=1,
    )
    ax.add_patch(shadow)

    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.18",
        linewidth=1.35,
        edgecolor=accent,
        facecolor=mix_with_white(accent, 0.92),
        zorder=2,
    )
    ax.add_patch(box)

    badge_circle = Circle((x + 0.34, y + h - 0.26), 0.16, facecolor=accent, edgecolor="none", zorder=4)
    ax.add_patch(badge_circle)
    ax.text(
        x + 0.34,
        y + h - 0.26,
        badge,
        ha="center",
        va="center",
        fontsize=9.8,
        color="white",
        fontweight="bold",
        zorder=5,
    )

    ax.text(
        x + 0.62,
        y + h - 0.25,
        title,
        ha="left",
        va="center",
        fontsize=10.6,
        color=PALETTE["ink"],
        fontweight="bold",
        zorder=5,
    )

    text_y = y + h - 0.56
    for line in lines:
        ax.text(
            x + 0.2,
            text_y,
            line,
            ha="left",
            va="top",
            fontsize=8.45,
            color=PALETTE["muted"],
            zorder=5,
        )
        text_y -= 0.29


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
            fontsize=7.8,
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
        "Existing Vision-Based Sign Recognition Workflow",
        ha="center",
        va="center",
        fontsize=21,
        color=PALETTE["ink"],
        fontweight="bold",
    )
    ax.text(
        8,
        9.04,
        "A typical image-first pipeline and the friction users feel during live sign interaction",
        ha="center",
        va="center",
        fontsize=11,
        color=PALETTE["muted"],
    )

    draw_pill(ax, 6.82, 8.38, "Typical existing pipeline", PALETTE["soft_blue"], PALETTE["blue"], PALETTE["blue"])
    draw_pill(ax, 6.98, 4.22, "Where the limitations appear", PALETTE["soft_coral"], PALETTE["coral"], PALETTE["coral"])

    card_y = 5.75
    card_w = 2.72
    card_h = 1.95
    xs = [0.55, 3.15, 5.75, 8.35, 10.95]

    draw_card(
        ax,
        xs[0],
        card_y,
        card_w,
        card_h,
        "Signer shows a gesture",
        [
            "A person places the hand in front of the webcam.",
            "The pose may still be settling into place.",
        ],
        PALETTE["coral"],
        "1",
        tag="human action",
    )
    draw_card(
        ax,
        xs[1],
        card_y,
        card_w,
        card_h,
        "Raw RGB frame captured",
        [
            "The frame includes the hand, clothes, background,",
            "lighting, and any other visual clutter nearby.",
        ],
        PALETTE["gold"],
        "2",
        tag="full image",
    )
    draw_card(
        ax,
        xs[2],
        card_y,
        card_w,
        card_h,
        "Image preprocessing",
        [
            "Resize, crop, or normalize the picture before",
            "sending pixels into the recognition model.",
        ],
        PALETTE["blue"],
        "3",
        tag="pixel cleanup",
    )
    draw_card(
        ax,
        xs[3],
        card_y,
        card_w,
        card_h,
        "CNN or vision model",
        [
            "The classifier predicts directly from image pixels.",
            "Accuracy can be good, but compute cost is higher.",
        ],
        PALETTE["teal"],
        "4",
        tag="image-based inference",
    )
    draw_card(
        ax,
        xs[4],
        card_y,
        card_w,
        card_h,
        "Immediate single-frame label",
        [
            "A letter is shown as soon as one frame is classified.",
            "There is usually no stability check before display.",
        ],
        PALETTE["slate"],
        "5",
        tag="instant output",
    )

    for idx in range(len(xs) - 1):
        draw_arrow(
            ax,
            (xs[idx] + card_w, card_y + 0.95),
            (xs[idx + 1], card_y + 0.95),
            PALETTE["muted"],
        )

    draw_limitation(
        ax,
        0.55,
        1.25,
        2.95,
        2.1,
        "Background-sensitive",
        [
            "Image appearance changes with lighting, clothing,",
            "camera angle, and clutter around the signer.",
        ],
        PALETTE["warning"],
        "L1",
    )
    draw_limitation(
        ax,
        3.65,
        1.25,
        2.95,
        2.1,
        "Heavier models",
        [
            "CNN-style pipelines often need more memory,",
            "more compute, and more time per prediction.",
        ],
        PALETTE["rose"],
        "L2",
    )
    draw_limitation(
        ax,
        6.75,
        1.25,
        2.95,
        2.1,
        "Prediction flicker",
        [
            "The shown label can jump while the hand is moving",
            "or while the user is still adjusting the pose.",
        ],
        PALETTE["coral"],
        "L3",
    )
    draw_limitation(
        ax,
        9.85,
        1.25,
        2.95,
        2.1,
        "Similar signs confuse",
        [
            "Letters such as A, S, E or M, N may look close,",
            "yet the system may still force one hard label.",
        ],
        PALETTE["gold"],
        "L4",
    )
    draw_limitation(
        ax,
        12.95,
        1.25,
        2.5,
        2.1,
        "Limited assistance",
        [
            "Many existing systems stop at the label and do not",
            "help with word building, hold logic, or speech.",
        ],
        PALETTE["slate"],
        "L5",
    )

    draw_arrow(ax, (4.52, 5.75), (2.05, 3.36), PALETTE["warning"], connection="arc3,rad=0.18")
    draw_arrow(ax, (9.72, 5.75), (5.15, 3.36), PALETTE["rose"], connection="arc3,rad=0.12")
    draw_arrow(ax, (12.3, 5.75), (8.25, 3.36), PALETTE["coral"], connection="arc3,rad=0.05")
    draw_arrow(ax, (9.72, 5.75), (11.35, 3.36), PALETTE["gold"], connection="arc3,rad=-0.12")
    draw_arrow(ax, (13.67, 5.75), (14.2, 3.36), PALETTE["slate"], connection="arc3,rad=-0.02")

    ax.text(
        8,
        0.34,
        "Typical result: workable recognition in controlled settings, but a less stable and less supportive live user experience.",
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
