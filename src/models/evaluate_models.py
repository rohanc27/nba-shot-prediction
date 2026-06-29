"""Evaluate trained models with calibration + probability buckets.

Usage:
    python -m src.models.evaluate_models
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

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
    "player_prior_shots",
    "player_prior_zone_shots",
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


def evaluate_model(name: str, model, X_test, y_test) -> dict:
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "auc": roc_auc_score(y_test, y_proba),
        "log_loss": log_loss(y_test, y_proba),
        "brier_score": brier_score_loss(y_test, y_proba),
    }

    print(f"\n=== {name} ===")
    for k, v in metrics.items():
        if k != "model":
            print(f"  {k}: {v:.4f}")

    return metrics, y_proba


def make_probability_buckets(y_test, y_proba, name: str) -> pd.DataFrame:
    df = pd.DataFrame({
        "y_true": y_test.values,
        "predicted_prob": y_proba,
    })

    df["bucket"] = pd.cut(
        df["predicted_prob"],
        bins=[i / 10 for i in range(11)],
        include_lowest=True,
    )

    buckets = df.groupby("bucket", observed=True).agg(
        n=("y_true", "count"),
        avg_predicted_prob=("predicted_prob", "mean"),
        actual_fg_pct=("y_true", "mean"),
    ).reset_index()

    buckets["model"] = name
    return buckets


def make_calibration_table(y_test, y_proba, name: str) -> pd.DataFrame:
    prob_true, prob_pred = calibration_curve(
        y_test,
        y_proba,
        n_bins=10,
        strategy="quantile",
    )

    return pd.DataFrame({
        "model": name,
        "mean_predicted_prob": prob_pred,
        "actual_fg_pct": prob_true,
    })


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(PROCESSED_DIR / "shots_features.parquet")
    test_df = df[df["SEASON"] == "2024-25"].copy()

    X_test = test_df[FEATURES]
    y_test = test_df[TARGET].astype(int)

    models = {
        "logreg": joblib.load(MODELS_DIR / "logreg.joblib"),
        "xgb": joblib.load(MODELS_DIR / "xgb.joblib"),
    }

    all_metrics = []
    all_buckets = []
    all_calibration = []

    for name, model in models.items():
        metrics, y_proba = evaluate_model(name, model, X_test, y_test)
        all_metrics.append(metrics)
        all_buckets.append(make_probability_buckets(y_test, y_proba, name))
        all_calibration.append(make_calibration_table(y_test, y_proba, name))

    metrics_df = pd.DataFrame(all_metrics)
    buckets_df = pd.concat(all_buckets, ignore_index=True)
    calibration_df = pd.concat(all_calibration, ignore_index=True)

    metrics_df.to_csv(REPORTS_DIR / "model_metrics.csv", index=False)
    buckets_df.to_csv(REPORTS_DIR / "probability_buckets.csv", index=False)
    calibration_df.to_csv(REPORTS_DIR / "calibration_table.csv", index=False)

    (REPORTS_DIR / "model_metrics.json").write_text(
        json.dumps(all_metrics, indent=2)
    )

    print("\nSaved:")
    print(f"  {REPORTS_DIR / 'model_metrics.csv'}")
    print(f"  {REPORTS_DIR / 'probability_buckets.csv'}")
    print(f"  {REPORTS_DIR / 'calibration_table.csv'}")


if __name__ == "__main__":
    main()
