"""Create comprehensive evaluation plots for trained models.

Usage:
    python -m src.visualization.model_dashboard
"""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

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
    "player_prior_zone_name_fg_pct",
    "player_prior_zone_range_fg_pct",
    "player_prior_action_fg_pct",
    "player_prior_2pt_pct",
    "player_prior_3pt_pct",
    "player_prior_shots",
    "player_prior_zone_shots",
    "player_prior_zone_name_shots",
    "player_prior_zone_range_shots",
    "player_prior_action_shots",
    "player_prior_2pt_shots",
    "player_prior_3pt_shots",
    "team_prior_fg_pct",
    "team_prior_zone_fg_pct",
    "team_prior_2pt_pct",
    "team_prior_3pt_pct",
    "team_prior_shots",
    "team_prior_zone_shots",
    "team_prior_2pt_shots",
    "team_prior_3pt_shots",
    "player_tendency_zone_rate",
    "player_tendency_action_rate",
    "player_tendency_shot_profile_rate",
    "player_zone_residual",
    "player_action_residual",
    "player_profile_residual",
    "is_home",
    "opponent_allowed_fg_pct",
    "opponent_allowed_zone_fg_pct",
    "opponent_allowed_2pt_pct",
    "opponent_allowed_3pt_pct",
    "opponent_allowed_shots",
    "opponent_allowed_zone_shots",
    "opponent_allowed_2pt_shots",
    "opponent_allowed_3pt_shots",
    "is_three",
    "is_corner_3",
    "is_layup",
    "is_dunk",
    "is_late_clock",
    "is_overtime",
    "action_category",
    "BASIC_ZONE",
    "shot_profile",
    "zone_range",
]

TARGET = "SHOT_MADE"


def load_test_data():
    df = pd.read_parquet(PROCESSED_DIR / "shots_features.parquet")
    test_df = df[df["SEASON"] == "2024-25"].copy()
    return test_df[FEATURES], test_df[TARGET].astype(int)


def load_models():
    return {
        "Logistic Regression": joblib.load(MODELS_DIR / "logreg.joblib"),
        "XGBoost": joblib.load(MODELS_DIR / "xgb.joblib"),
        "Tuned XGBoost": joblib.load(MODELS_DIR / "xgb_tuned.joblib"),
    }


def plot_roc(models, X_test, y_test):
    plt.figure(figsize=(7, 6))

    for name, model in models.items():
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        plt.plot(fpr, tpr, label=f"{name} AUC={auc:.4f}")

    plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "roc_curve.png", dpi=200)
    plt.close()


def plot_precision_recall(models, X_test, y_test):
    plt.figure(figsize=(7, 6))

    for name, model in models.items():
        y_proba = model.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        ap = average_precision_score(y_test, y_proba)
        plt.plot(recall, precision, label=f"{name} AP={ap:.4f}")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "pr_curve.png", dpi=200)
    plt.close()


def plot_confusion_matrices(models, X_test, y_test):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    for ax, (name, model) in zip(axes, models.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["Miss", "Make"],
        )
        disp.plot(ax=ax, values_format=",d", colorbar=False)
        ax.set_title(name)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "confusion_matrices.png", dpi=200)
    plt.close()


def make_lift_data(y_true, y_proba, n_bins=10):
    df = pd.DataFrame({"y_true": y_true, "y_proba": y_proba})
    df = df.sort_values("y_proba", ascending=False).reset_index(drop=True)
    df["decile"] = pd.qcut(df.index + 1, q=n_bins, labels=False)

    lift = df.groupby("decile").agg(
        n=("y_true", "count"),
        actual_fg_pct=("y_true", "mean"),
        avg_predicted_prob=("y_proba", "mean"),
    ).reset_index()

    baseline = df["y_true"].mean()
    lift["lift"] = lift["actual_fg_pct"] / baseline
    lift["decile"] = lift["decile"] + 1
    return lift


def plot_lift(models, X_test, y_test):
    plt.figure(figsize=(8, 5))

    for name, model in models.items():
        y_proba = model.predict_proba(X_test)[:, 1]
        lift = make_lift_data(y_test, y_proba)
        plt.plot(lift["decile"], lift["lift"], marker="o", label=name)

    plt.axhline(1.0, linestyle="--", label="Baseline")
    plt.xlabel("Prediction decile, highest probability first")
    plt.ylabel("Lift over average FG%")
    plt.title("Lift Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "lift_curve.png", dpi=200)
    plt.close()


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading test data...")
    X_test, y_test = load_test_data()

    print("Loading models...")
    models = load_models()

    print("Creating ROC curve...")
    plot_roc(models, X_test, y_test)

    print("Creating precision-recall curve...")
    plot_precision_recall(models, X_test, y_test)

    print("Creating confusion matrices...")
    plot_confusion_matrices(models, X_test, y_test)

    print("Creating lift curve...")
    plot_lift(models, X_test, y_test)

    print("\nSaved:")
    print(f"  {FIGURES_DIR / 'roc_curve.png'}")
    print(f"  {FIGURES_DIR / 'pr_curve.png'}")
    print(f"  {FIGURES_DIR / 'confusion_matrices.png'}")
    print(f"  {FIGURES_DIR / 'lift_curve.png'}")


if __name__ == "__main__":
    main()
