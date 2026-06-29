"""Train an XGBoost classifier for shot make/miss prediction.

Same train/test split as the logreg baseline (temporal split by season).
Saves model + preprocessor pipeline to models/xgb.joblib.

Usage:
    python -m src.models.train_xgb
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

NUMERIC_FEATURES = [
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
]

BINARY_FEATURES = [
    "is_three",
    "is_corner_3",
    "is_layup",
    "is_dunk",
    "is_late_clock",
    "is_overtime",
]

CATEGORICAL_FEATURES = [
    "action_category",
    "BASIC_ZONE",
]

TARGET = "SHOT_MADE"


def load_data(features_path: Path):
    """Load and split into train/test by season."""
    df = pd.read_parquet(features_path)
    print(f"Loaded {len(df):,} shots")

    train_seasons = ["2022-23", "2023-24"]
    test_seasons = ["2024-25"]

    train_df = df[df["SEASON"].isin(train_seasons)].copy()
    test_df = df[df["SEASON"].isin(test_seasons)].copy()

    print(f"  Train: {len(train_df):,} shots from {train_seasons}")
    print(f"  Test:  {len(test_df):,} shots from {test_seasons}")
    return train_df, test_df


def build_preprocessor() -> ColumnTransformer:
    """Build a sklearn ColumnTransformer.

    XGBoost doesn't need scaling, so numeric features pass through.
    Categorical features get one-hot encoded.
    """
    return ColumnTransformer(
        transformers=[
            ("numeric", "passthrough", NUMERIC_FEATURES),
            ("binary", "passthrough", BINARY_FEATURES),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )


def evaluate(y_true, y_pred, y_proba) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "auc": float(roc_auc_score(y_true, y_proba)),
        "log_loss": float(log_loss(y_true, y_proba)),
        "brier_score": float(brier_score_loss(y_true, y_proba)),
        "n_samples": int(len(y_true)),
        "baseline_predict_majority": float(
            max(y_true.mean(), 1 - y_true.mean())
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features",
        type=Path,
        default=PROCESSED_DIR / "shots_features.parquet",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=MODELS_DIR / "xgb.joblib",
    )
    args = parser.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    train_df, test_df = load_data(args.features)

    all_features = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES
    X_train = train_df[all_features]
    y_train = train_df[TARGET].astype(int)
    X_test = test_df[all_features]
    y_test = test_df[TARGET].astype(int)

    pipeline = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", xgb.XGBClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_weight=5,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
            tree_method="hist",
        )),
    ])

    print("\nFitting XGBoost (this takes ~30-60 seconds)...")
    pipeline.fit(X_train, y_train)

    print("\nEvaluating...")
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    metrics = evaluate(y_test, y_pred, y_proba)

    print("\n=== Test metrics (2024-25 season) ===")
    print(f"  Accuracy:       {metrics['accuracy']:.4f}")
    print(f"  AUC:            {metrics['auc']:.4f}")
    print(f"  Log loss:       {metrics['log_loss']:.4f}")
    print(f"  Brier score:    {metrics['brier_score']:.4f}")
    print(f"  Baseline:       {metrics['baseline_predict_majority']:.4f}")
    print(f"  N test samples: {metrics['n_samples']:,}")

    # Feature importance
    print("\n=== Top 15 features by importance ===")
    classifier = pipeline.named_steps["classifier"]
    preprocessor = pipeline.named_steps["preprocessor"]
    feature_names = preprocessor.get_feature_names_out()
    importances = classifier.feature_importances_
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False).head(15)
    print(importance_df.to_string(index=False))

    joblib.dump(pipeline, args.out)
    print(f"\nModel saved to {args.out}")

    metrics_path = args.out.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
