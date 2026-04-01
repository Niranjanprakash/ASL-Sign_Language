"""
Generate a simple, paper-friendly architecture diagram for the ASL project.

Exports:
  - ../Architecture.png
  - ./plots/simple_architecture_diagram.png
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
PLOTS_DIR = Path(__file__).resolve().parent / "plots"
OUTPUTS = [
    ROOT / "Architecture.png",
    PLOTS_DIR / "simple_architecture_diagram.png",
]


def add_box(ax, x, y, w, h, text, fontsize=10.5, bold=False):
    shadow = FancyBboxPatch(
        (x + 0.05, y - 0.05),
        w,
        h,
        boxstyle="square,pad=0.02",
        linewidth=0,
        facecolor="#cfcfcf",
        alpha=0.45,
        zorder=1,
    )
    ax.add_patch(shadow)

    rect = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="square,pad=0.02",
        linewidth=1.0,
        edgecolor="#b5b5b5",
        facecolor="#f8f8f8",
        zorder=2,
    )
    ax.add_patch(rect)

    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color="#333333",
        fontweight="bold" if bold else "normal",
        zorder=3,
    )


def add_arrow(ax, start, end):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=11,
        linewidth=1.0,
        color="#606060",
        zorder=4,
    )
    ax.add_patch(arrow)


def main():
    fig, ax = plt.subplots(figsize=(9, 6.8), dpi=220)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 11)
    ax.axis("off")

    # Left inputs
    add_box(ax, 0.25, 8.25, 2.8, 0.95, "Webcam Camera")
    add_box(ax, 0.25, 6.2, 2.8, 0.95, "Webcam Frame Capture")
    add_box(ax, 0.25, 4.15, 2.8, 0.95, "Image Preprocessing\nResize + RGB Conversion")

    # Center engine
    engine_text = (
        "ASL Recognition\n"
        "Processing Engine\n\n"
        "(MediaPipe Hands Landmark Detection\n"
        "+ Feature Processing)\n\n"
        "• Hand Detection\n"
        "• 21 Hand Landmarks\n"
        "• 63 Raw Landmark Values\n"
        "• Feature Extraction\n"
        "• Feature Normalization\n"
        "• 93 Input Features"
    )
    add_box(ax, 5.0, 3.2, 3.0, 6.6, engine_text, fontsize=9.0, bold=False)

    # Right outputs
    add_box(ax, 9.2, 8.55, 3.2, 0.85, "Hand Landmark\nVisualization", fontsize=9.0)
    add_box(ax, 9.2, 6.9, 3.2, 0.85, "Gesture Feature Analyzer", fontsize=9.0)
    add_box(ax, 9.2, 5.25, 3.2, 0.85, "Neural Network Classifier", fontsize=9.0)
    add_box(ax, 9.2, 3.6, 3.2, 0.85, "Prediction Confidence", fontsize=9.0)

    # Bottom classifier
    classifier_text = (
        "Deep Neural Network Classifier\n\n"
        "Input (93)\n"
        "Dense (128) + ReLU\n"
        "Dense (64) + ReLU\n"
        "Softmax (29 Classes)"
    )
    add_box(ax, 4.2, 1.25, 4.6, 1.45, classifier_text, fontsize=9.0)

    # Final output
    output_text = (
        "ASL Prediction Output\n"
        "Predicted Class (A-Z, del, space, nothing)\n"
        "Confidence Score"
    )
    add_box(ax, 4.6, 0.05, 3.8, 0.82, output_text, fontsize=8.6)

    # Arrows from left to center
    add_arrow(ax, (3.05, 8.72), (5.0, 8.72))
    add_arrow(ax, (3.05, 6.67), (5.0, 6.67))
    add_arrow(ax, (3.05, 4.62), (5.0, 4.62))

    # Arrows from center to right
    add_arrow(ax, (8.0, 9.05), (9.2, 9.05))
    add_arrow(ax, (8.0, 7.32), (9.2, 7.32))
    add_arrow(ax, (8.0, 5.67), (9.2, 5.67))
    add_arrow(ax, (8.0, 4.02), (9.2, 4.02))

    # Vertical flow
    add_arrow(ax, (6.5, 2.7), (6.5, 3.2))
    add_arrow(ax, (6.5, 1.25), (6.5, 0.87))

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    for output_path in OUTPUTS:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight", pad_inches=0.12, facecolor=fig.get_facecolor())

    plt.close(fig)
    print("Saved outputs:")
    for output_path in OUTPUTS:
        print(f"- {output_path}")


if __name__ == "__main__":
    main()
