"""Tune XGBoost hyperparameters with RandomizedSearchCV.

Uses 2022-23 as train, 2023-24 as validation during tuning.
Final evaluation remains untouched on 2024-25.

Usage:
    python -m src.models.tune_xgb
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import xgboost as xgb
from scipy.stats import randint, uniform
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, PredefinedSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

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
    "player_tendency_zone_rate",
    "player_tendency_action_rate",
    "player_tendency_shot_profile_rate",
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
    "shot_profile",
    "zone_range",
]

TARGET = "SHOT_MADE"


def build_preprocessor() -> ColumnTransformer:
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


def evaluate(y_true, y_proba) -> dict:
    y_pred = (y_proba >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "auc": float(roc_auc_score(y_true, y_proba)),
        "log_loss": float(log_loss(y_true, y_proba)),
        "brier_score": float(brier_score_loss(y_true, y_proba)),
        "n_samples": int(len(y_true)),
        "baseline_predict_majority": float(max(y_true.mean(), 1 - y_true.mean())),
    }


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading features...")
    df = pd.read_parquet(PROCESSED_DIR / "shots_features.parquet")

    tune_df = df[df["SEASON"].isin(["2022-23", "2023-24"])].copy()
    test_df = df[df["SEASON"] == "2024-25"].copy()

    all_features = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES

    X_tune = tune_df[all_features]
    y_tune = tune_df[TARGET].astype(int)

    X_test = test_df[all_features]
    y_test = test_df[TARGET].astype(int)

    # PredefinedSplit:
    # -1 means always train fold
    #  0 means validation fold
    # So we tune on 2022-23 -> validate on 2023-24.
    split_index = tune_df["SEASON"].map({
        "2022-23": -1,
        "2023-24": 0,
    }).to_numpy()

    cv = PredefinedSplit(test_fold=split_index)

    pipeline = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
            tree_method="hist",
        )),
    ])

    param_distributions = {
        "classifier__n_estimators": randint(250, 900),
        "classifier__max_depth": randint(2, 8),
        "classifier__learning_rate": uniform(0.015, 0.12),
        "classifier__subsample": uniform(0.65, 0.35),
        "classifier__colsample_bytree": uniform(0.65, 0.35),
        "classifier__min_child_weight": randint(1, 20),
        "classifier__gamma": uniform(0.0, 4.0),
        "classifier__reg_alpha": uniform(0.0, 1.0),
        "classifier__reg_lambda": uniform(0.5, 5.0),
    }

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=30,
        scoring="roc_auc",
        cv=cv,
        verbose=2,
        random_state=42,
        n_jobs=1,
        refit=True,
    )

    print("\nTuning XGBoost...")
    search.fit(X_tune, y_tune)

    print("\n=== Best validation result ===")
    print(f"Best validation AUC: {search.best_score_:.4f}")
    print("Best params:")
    for key, value in search.best_params_.items():
        print(f"  {key}: {value}")

    print("\nEvaluating best model on untouched 2024-25 test set...")
    best_model = search.best_estimator_
    y_proba = best_model.predict_proba(X_test)[:, 1]
    metrics = evaluate(y_test, y_proba)

    print("\n=== Tuned XGBoost test metrics (2024-25) ===")
    print(f"  Accuracy:    {metrics['accuracy']:.4f}")
    print(f"  AUC:         {metrics['auc']:.4f}")
    print(f"  Log loss:    {metrics['log_loss']:.4f}")
    print(f"  Brier score: {metrics['brier_score']:.4f}")
    print(f"  Baseline:    {metrics['baseline_predict_majority']:.4f}")
    print(f"  N samples:   {metrics['n_samples']:,}")

    out_model = MODELS_DIR / "xgb_tuned.joblib"
    joblib.dump(best_model, out_model)

    results = {
        "best_validation_auc": float(search.best_score_),
        "best_params": search.best_params_,
        "test_metrics": metrics,
    }

    out_metrics = REPORTS_DIR / "xgb_tuned_results.json"
    out_metrics.write_text(json.dumps(results, indent=2))

    cv_results = pd.DataFrame(search.cv_results_)
    cv_results.to_csv(REPORTS_DIR / "xgb_tuning_cv_results.csv", index=False)

    print("\nSaved:")
    print(f"  {out_model}")
    print(f"  {out_metrics}")
    print(f"  {REPORTS_DIR / 'xgb_tuning_cv_results.csv'}")


if __name__ == "__main__":
    main()
