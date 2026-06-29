"""Plot model calibration curves and probability buckets.

Usage:
    python -m src.visualization.plot_calibration
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    calibration = pd.read_csv(REPORTS_DIR / "calibration_table.csv")
    buckets = pd.read_csv(REPORTS_DIR / "probability_buckets.csv")

    plt.figure(figsize=(7, 6))
    for model_name, group in calibration.groupby("model"):
        plt.plot(
            group["mean_predicted_prob"],
            group["actual_fg_pct"],
            marker="o",
            label=model_name,
        )

    plt.plot([0, 1], [0, 1], linestyle="--", label="perfect calibration")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Actual FG%")
    plt.title("Calibration Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "calibration_curve.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    for model_name, group in buckets.groupby("model"):
        plt.plot(
            group["avg_predicted_prob"],
            group["actual_fg_pct"],
            marker="o",
            label=model_name,
        )

    plt.plot([0, 1], [0, 1], linestyle="--", label="perfect calibration")
    plt.xlabel("Average predicted probability bucket")
    plt.ylabel("Actual FG%")
    plt.title("Probability Bucket Calibration")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "probability_bucket_calibration.png", dpi=200)
    plt.close()

    print("Saved:")
    print(f"  {FIGURES_DIR / 'calibration_curve.png'}")
    print(f"  {FIGURES_DIR / 'probability_bucket_calibration.png'}")


if __name__ == "__main__":
    main()
