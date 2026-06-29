"""Generate project architecture diagrams.

Usage:
    python -m src.docs.generate_architecture
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOCS_DIR = PROJECT_ROOT / "docs"


def draw_box(ax, x, y, w, h, text, fc="#1B2630", ec="#F2AA4C"):
    box = patches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.03,rounding_size=0.08",
        facecolor=fc,
        edgecolor=ec,
        linewidth=1.6,
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        color="white",
        fontsize=11,
        fontweight="bold",
        wrap=True,
    )


def arrow(ax, x1, y1, x2, y2):
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(arrowstyle="->", color="#D6D6D6", lw=2),
    )


def setup_ax(title):
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 8)
    ax.axis("off")
    ax.set_facecolor("#101820")
    fig.patch.set_facecolor("#101820")
    ax.text(
        7.5,
        7.45,
        title,
        color="white",
        fontsize=24,
        fontweight="bold",
        ha="center",
    )
    return fig, ax


def architecture():
    fig, ax = setup_ax("End-to-End ML Architecture")

    boxes = [
        (0.7, 5.2, 2.2, 1.0, "Raw NBA\nShot Data"),
        (3.4, 5.2, 2.2, 1.0, "Cleaning &\nFiltering"),
        (6.1, 5.2, 2.2, 1.0, "Feature\nEngineering"),
        (8.8, 5.2, 2.2, 1.0, "Model\nTraining"),
        (11.5, 5.2, 2.2, 1.0, "Evaluation &\nExplainability"),
        (6.1, 2.8, 2.2, 1.0, "Final XGBoost\nModel"),
        (8.8, 2.8, 2.2, 1.0, "Streamlit\nApp"),
        (11.5, 2.8, 2.2, 1.0, "Interactive\nxFG Tool"),
    ]

    for b in boxes:
        draw_box(ax, *b)

    arrow(ax, 2.9, 5.7, 3.4, 5.7)
    arrow(ax, 5.6, 5.7, 6.1, 5.7)
    arrow(ax, 8.3, 5.7, 8.8, 5.7)
    arrow(ax, 11.0, 5.7, 11.5, 5.7)
    arrow(ax, 9.9, 5.2, 7.2, 3.8)
    arrow(ax, 8.3, 3.3, 8.8, 3.3)
    arrow(ax, 11.0, 3.3, 11.5, 3.3)

    out = DOCS_DIR / "architecture.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def feature_pipeline():
    fig, ax = setup_ax("Feature Engineering Pipeline")

    center_x = 7.5
    draw_box(ax, center_x - 1.25, 6.2, 2.5, 0.85, "Shot Attempt")

    feature_boxes = [
        (0.8, 4.4, 2.4, 1.0, "Shot Geometry\nDistance, angle, location"),
        (3.5, 4.4, 2.4, 1.0, "Player History\nFG%, zone, action"),
        (6.2, 4.4, 2.4, 1.0, "Team Context\nOffensive priors"),
        (8.9, 4.4, 2.4, 1.0, "Opponent Context\nAllowed FG%"),
        (11.6, 4.4, 2.4, 1.0, "Shot Profiles\nZone × range × action"),
        (3.5, 2.3, 2.4, 1.0, "Tendencies\nTypical player shots"),
        (6.2, 2.3, 2.4, 1.0, "Residual Skill\nAbove/below league avg"),
        (8.9, 2.3, 2.4, 1.0, "Game State\nClock, quarter, home"),
    ]

    for b in feature_boxes:
        draw_box(ax, *b)

    draw_box(ax, center_x - 1.4, 0.7, 2.8, 0.9, "XGBoost xFG Model", fc="#243B53")

    for x, y, w, h, _ in feature_boxes:
        arrow(ax, center_x, 6.2, x + w / 2, y + h)
        arrow(ax, x + w / 2, y, center_x, 1.6)

    out = DOCS_DIR / "feature_pipeline.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def training_pipeline():
    fig, ax = setup_ax("Model Development Workflow")

    boxes = [
        (0.8, 4.8, 2.2, 1.0, "Processed\nFeatures"),
        (3.5, 4.8, 2.2, 1.0, "Temporal\nSplit"),
        (6.2, 5.7, 2.2, 1.0, "Logistic\nRegression"),
        (6.2, 3.9, 2.2, 1.0, "XGBoost\nBaseline"),
        (8.9, 3.9, 2.2, 1.0, "Randomized\nSearch CV"),
        (11.6, 3.9, 2.2, 1.0, "Final Tuned\nXGBoost"),
        (11.6, 1.8, 2.2, 1.0, "Frozen Model\nxgb_final.joblib"),
    ]

    for b in boxes:
        draw_box(ax, *b)

    arrow(ax, 3.0, 5.3, 3.5, 5.3)
    arrow(ax, 5.7, 5.3, 6.2, 6.2)
    arrow(ax, 5.7, 5.3, 6.2, 4.4)
    arrow(ax, 8.4, 4.4, 8.9, 4.4)
    arrow(ax, 11.1, 4.4, 11.6, 4.4)
    arrow(ax, 12.7, 3.9, 12.7, 2.8)

    ax.text(
        4.6,
        4.15,
        "Train: 2022-23, 2023-24\nTest: 2024-25",
        color="#D6D6D6",
        fontsize=11,
        ha="center",
    )

    out = DOCS_DIR / "training_pipeline.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def main() -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    architecture()
    feature_pipeline()
    training_pipeline()


if __name__ == "__main__":
    main()
