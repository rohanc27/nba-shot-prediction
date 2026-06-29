"""Detailed error analysis for tuned XGBoost model.

Usage:
    python -m src.models.error_analysis
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score

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
]

TARGET = "SHOT_MADE"

DISPLAY_COLS = [
    "GAME_DATE",
    "PLAYER_NAME",
    "TEAM_NAME",
    "HOME_TEAM",
    "AWAY_TEAM",
    "ACTION_TYPE",
    "action_category",
    "SHOT_TYPE",
    "BASIC_ZONE",
    "ZONE_NAME",
    "ZONE_RANGE",
    "SHOT_DISTANCE",
    "LOC_X",
    "LOC_Y",
    "QUARTER",
    "MINS_LEFT",
    "SECS_LEFT",
    "SHOT_MADE",
    "predicted_prob",
    "prediction",
    "absolute_error",
]


def load_predictions() -> pd.DataFrame:
    df = pd.read_parquet(PROCESSED_DIR / "shots_features.parquet")
    test_df = df[df["SEASON"] == "2024-25"].copy()

    model = joblib.load(MODELS_DIR / "xgb_tuned.joblib")

    X_test = test_df[FEATURES]
    y_true = test_df[TARGET].astype(int)

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    test_df["predicted_prob"] = y_proba
    test_df["prediction"] = y_pred
    test_df["correct"] = y_pred == y_true
    test_df["absolute_error"] = (y_true - y_proba).abs()

    return test_df


def save_worst_examples(df: pd.DataFrame) -> None:
    # Model was confident shot would be made, but it missed.
    false_positives = df[
        (df["prediction"] == 1) & (df["SHOT_MADE"] == 0)
    ].sort_values("predicted_prob", ascending=False)

    false_positives[DISPLAY_COLS].head(100).to_csv(
        REPORTS_DIR / "worst_false_positives.csv",
        index=False,
    )

    # Model was confident shot would miss, but it went in.
    false_negatives = df[
        (df["prediction"] == 0) & (df["SHOT_MADE"] == 1)
    ].sort_values("predicted_prob", ascending=True)

    false_negatives[DISPLAY_COLS].head(100).to_csv(
        REPORTS_DIR / "worst_false_negatives.csv",
        index=False,
    )


def summarize_group(df: pd.DataFrame, group_col: str, min_attempts: int = 0):
    rows = []

    for value, group in df.groupby(group_col):
        if len(group) < min_attempts:
            continue

        rows.append({
            group_col: value,
            "n": len(group),
            "accuracy": accuracy_score(group["SHOT_MADE"], group["prediction"]),
            "avg_probability": group["predicted_prob"].mean(),
            "actual_fg_pct": group["SHOT_MADE"].mean(),
            "avg_absolute_error": group["absolute_error"].mean(),
        })

    return pd.DataFrame(rows).sort_values("avg_absolute_error", ascending=False)


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading predictions...")
    df = load_predictions()

    print("Saving worst examples...")
    save_worst_examples(df)

    print("Saving zone error report...")
    summarize_group(df, "BASIC_ZONE").to_csv(
        REPORTS_DIR / "error_by_zone.csv",
        index=False,
    )

    print("Saving action error report...")
    summarize_group(df, "action_category").to_csv(
        REPORTS_DIR / "error_by_action.csv",
        index=False,
    )

    print("Saving player error report...")
    summarize_group(df, "PLAYER_NAME", min_attempts=200).to_csv(
        REPORTS_DIR / "error_by_player.csv",
        index=False,
    )

    print("\nSaved:")
    print(f"  {REPORTS_DIR / 'worst_false_positives.csv'}")
    print(f"  {REPORTS_DIR / 'worst_false_negatives.csv'}")
    print(f"  {REPORTS_DIR / 'error_by_zone.csv'}")
    print(f"  {REPORTS_DIR / 'error_by_action.csv'}")
    print(f"  {REPORTS_DIR / 'error_by_player.csv'}")


if __name__ == "__main__":
    main()
