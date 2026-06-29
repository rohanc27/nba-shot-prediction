"""Generate README banner image.

Usage:
    python -m src.docs.generate_banner
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOCS_DIR = PROJECT_ROOT / "docs"


def main() -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 5)
    ax.axis("off")

    # Background
    ax.add_patch(
        patches.Rectangle(
            (0, 0),
            16,
            5,
            color="#101820",
        )
    )

    # Court-inspired lines
    ax.add_patch(
        patches.Rectangle(
            (10.4, 0.5),
            4.8,
            4.0,
            fill=False,
            edgecolor="#F2AA4C",
            linewidth=2,
            alpha=0.7,
        )
    )
    ax.add_patch(
        patches.Circle(
            (12.8, 2.5),
            1.0,
            fill=False,
            edgecolor="#F2AA4C",
            linewidth=2,
            alpha=0.7,
        )
    )
    ax.add_patch(
        patches.Arc(
            (12.8, 0.7),
            4.2,
            4.2,
            theta1=20,
            theta2=160,
            edgecolor="#F2AA4C",
            linewidth=2,
            alpha=0.7,
        )
    )

    # Text
    ax.text(
        0.8,
        3.35,
        "🏀 NBA Expected Field Goal Model",
        fontsize=34,
        fontweight="bold",
        color="white",
        va="center",
    )
    ax.text(
        0.85,
        2.55,
        "Predicting NBA shot make probability with shot geometry, historical priors, XGBoost, and Streamlit",
        fontsize=16,
        color="#D6D6D6",
        va="center",
    )

    # Metric cards
    cards = [
        ("654K+", "Shots"),
        ("0.658", "Test AUC"),
        ("0.640", "Log Loss"),
        ("xFG", "Model"),
    ]

    x = 0.85
    for value, label in cards:
        ax.add_patch(
            patches.FancyBboxPatch(
                (x, 0.75),
                2.25,
                0.95,
                boxstyle="round,pad=0.02,rounding_size=0.08",
                facecolor="#1B2630",
                edgecolor="#F2AA4C",
                linewidth=1.4,
                alpha=0.95,
            )
        )
        ax.text(
            x + 1.125,
            1.28,
            value,
            fontsize=22,
            color="white",
            fontweight="bold",
            ha="center",
            va="center",
        )
        ax.text(
            x + 1.125,
            0.93,
            label,
            fontsize=10,
            color="#D6D6D6",
            ha="center",
            va="center",
        )
        x += 2.55

    out = DOCS_DIR / "banner.png"
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
