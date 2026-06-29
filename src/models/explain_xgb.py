"""Explain the trained XGBoost model with SHAP.

Usage:
    python -m src.models.explain_xgb
"""

from pathlib import Path

import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

FEATURES = [
    "SHOT_DISTANCE",
    "LOC_X",
    "LOC_Y",
    "shot_angle_deg",
    "abs_angle_deg",
    "seconds_remaining_in_quarter",
    "QUARTER",
    "player_prior_fg_pct",
    "player_prior_zone_fg_pct",
    "player_prior_action_fg_pct",
    "player_prior_2pt_pct",
    "player_prior_3pt_pct",
    "player_prior_shots",
    "player_prior_zone_shots",
    "player_prior_action_shots",
    "player_prior_2pt_shots",
    "player_prior_3pt_shots",
    "is_three",
    "is_corner_3",
    "is_layup",
    "is_dunk",
    "is_late_clock",
    "is_overtime",
    "action_category",
    "BASIC_ZONE",
]

TARGET = "SHOT_MADE"


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data/model...")
    df = pd.read_parquet(PROCESSED_DIR / "shots_features.parquet")
    test_df = df[df["SEASON"] == "2024-25"].copy()

    # SHAP can be slow, so sample the test set
    sample_df = test_df.sample(n=5000, random_state=42)

    X_sample_raw = sample_df[FEATURES]
    y_sample = sample_df[TARGET].astype(int)

    pipeline = joblib.load(MODELS_DIR / "xgb.joblib")
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]

    print("Transforming features...")
    X_sample = preprocessor.transform(X_sample_raw)
    feature_names = preprocessor.get_feature_names_out()

    X_sample = pd.DataFrame(X_sample, columns=feature_names)

    print("Computing SHAP values...")
    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(X_sample)

    print("Saving SHAP summary plot...")
    shap.summary_plot(
        shap_values,
        X_sample,
        show=False,
        max_display=20,
    )
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_summary.png", dpi=200, bbox_inches="tight")
    plt.close()

    print("Saving SHAP bar plot...")
    shap.summary_plot(
        shap_values,
        X_sample,
        plot_type="bar",
        show=False,
        max_display=20,
    )
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_bar.png", dpi=200, bbox_inches="tight")
    plt.close()

    print("Saving SHAP values table...")
    shap_importance = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": abs(shap_values).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False)

    shap_importance.to_csv(REPORTS_DIR / "shap_feature_importance.csv", index=False)

    print("\n=== Top 20 SHAP features ===")
    print(shap_importance.head(20).to_string(index=False))

    print("\nSaved:")
    print(f"  {FIGURES_DIR / 'shap_summary.png'}")
    print(f"  {FIGURES_DIR / 'shap_bar.png'}")
    print(f"  {REPORTS_DIR / 'shap_feature_importance.csv'}")


if __name__ == "__main__":
    main()
