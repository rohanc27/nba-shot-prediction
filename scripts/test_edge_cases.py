"""Edge-case comparison: pre-distance-bucketing model vs. current model.

The frozen "before" models were overwritten during retraining, so this
script reconstructs the old feature set from git history (the commit
immediately before distance bucketing/stratified priors were added) and
trains a throwaway in-memory copy on the SAME train split, purely for a
fair side-by-side comparison on the SAME real shot rows. Nothing here is
saved as a production model.
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# --- OLD feature set (as of commit bf912db, before this task's changes) ---
OLD_NUMERIC = [
    "SHOT_DISTANCE", "LOC_X", "LOC_Y", "shot_angle_deg", "abs_angle_deg",
    "seconds_remaining_in_quarter", "QUARTER",
    "player_prior_fg_pct", "player_prior_zone_fg_pct", "player_prior_action_fg_pct",
    "player_prior_2pt_pct", "player_prior_3pt_pct", "player_prior_shots",
    "player_prior_zone_shots", "player_prior_action_shots",
    "player_prior_2pt_shots", "player_prior_3pt_shots",
    "team_prior_fg_pct", "team_prior_zone_fg_pct", "team_prior_2pt_pct",
    "team_prior_3pt_pct", "team_prior_shots", "team_prior_zone_shots",
    "team_prior_2pt_shots", "team_prior_3pt_shots", "is_home",
    "opponent_allowed_fg_pct", "opponent_allowed_zone_fg_pct",
    "opponent_allowed_2pt_pct", "opponent_allowed_3pt_pct",
    "opponent_allowed_shots", "opponent_allowed_zone_shots",
    "opponent_allowed_2pt_shots", "opponent_allowed_3pt_shots",
    "player_tendency_zone_rate", "player_tendency_action_rate",
    "player_tendency_shot_profile_rate", "player_zone_residual",
    "player_action_residual", "player_profile_residual",
    "proxy_defender_dist_ft", "proxy_shot_clock_sec", "proxy_dribbles",
    "proxy_touch_time_sec", "height_inches", "wingspan_inches",
    "wingspan_minus_height_in",
]
OLD_BINARY = ["is_three", "is_corner_3", "is_layup", "is_dunk", "is_late_clock", "is_overtime"]
CATEGORICAL = ["action_category", "BASIC_ZONE", "shot_profile", "zone_range"]
TARGET = "SHOT_MADE"

# --- NEW feature set adds distance buckets + distance-stratified priors ---
DIST_BUCKET_BINARY = [
    "distance_0_10_ft", "distance_10_16_ft", "distance_16_24_ft",
    "distance_24_28_ft", "distance_28_35_ft", "distance_35_plus_ft",
]
DIST_PRIOR_NUMERIC = [
    f"player_prior_{stat}_{b}"
    for b in ["0_10", "10_16", "16_24", "24_28", "28_35", "35_plus"]
    for stat in ["fg_pct", "shots"]
]
NEW_NUMERIC = OLD_NUMERIC + DIST_PRIOR_NUMERIC
NEW_BINARY = OLD_BINARY + DIST_BUCKET_BINARY


def build_pipeline(numeric, binary):
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric),
            ("binary", "passthrough", binary),
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ],
        remainder="drop",
    )
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000, n_jobs=-1, random_state=42)),
    ])


def build_xgb_pipeline(numeric, binary):
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", "passthrough", numeric),
            ("binary", "passthrough", binary),
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ],
        remainder="drop",
    )
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", xgb.XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            subsample=0.85, colsample_bytree=0.85, min_child_weight=5,
            objective="binary:logistic", eval_metric="logloss",
            random_state=42, n_jobs=-1, tree_method="hist",
        )),
    ])


def main() -> None:
    df = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "shots_features.parquet")
    train_df = df[df["SEASON"].isin(["2022-23", "2023-24"])].copy()
    test_df = df[df["SEASON"] == "2024-25"].copy()

    print("Training OLD-style (pre-distance-bucketing) LogReg in-memory for comparison...")
    old_logreg = build_pipeline(OLD_NUMERIC, OLD_BINARY)
    old_logreg.fit(train_df[OLD_NUMERIC + OLD_BINARY + CATEGORICAL], train_df[TARGET].astype(int))

    print("Training OLD-style (pre-distance-bucketing) XGBoost in-memory for comparison...")
    old_xgb = build_xgb_pipeline(OLD_NUMERIC, OLD_BINARY)
    old_xgb.fit(train_df[OLD_NUMERIC + OLD_BINARY + CATEGORICAL], train_df[TARGET].astype(int))

    print("Loading the actual saved NEW models (models/logreg.joblib, models/xgb.joblib)...")
    new_logreg = joblib.load(PROJECT_ROOT / "models" / "logreg.joblib")
    new_xgb_model = joblib.load(PROJECT_ROOT / "models" / "xgb.joblib")

    # NB: must slice with .loc[[idx]] (a list), not .loc[idx] / .iloc[0], to
    # keep a proper 2D DataFrame with per-column dtypes preserved -- a
    # Series-based single-row extraction (.iloc[0].to_frame().T) casts every
    # column to dtype=object and silently produces wrong probabilities.
    # Caught this by cross-checking against the saved model's batch-mode
    # calibration (0-10ft: 58.0% pred vs 57.6% actual, 35+ft: 11.7% pred vs
    # 11.2% actual -- both sane) before trusting any single-row number here.
    def predict(model, numeric, binary, idx, overrides=None):
        row_df = test_df.loc[[idx], numeric + binary + CATEGORICAL].copy()
        if overrides:
            for k, v in overrides.items():
                if k in row_df.columns:
                    row_df[k] = v
        return model.predict_proba(row_df)[0, 1]

    cases = []

    # Case 1: Curry's real deepest 2024-25 shot (40ft, actually missed)
    curry_40_idx = test_df[(test_df["PLAYER_NAME"] == "Stephen Curry") & (test_df["SHOT_DISTANCE"] == 40)].index[0]
    cases.append(("Curry, real 40ft heave (missed)", curry_40_idx, None, "REAL"))

    # Case 1b: synthetic 50ft heave -- override distance fields on the 40ft
    # row, since no real 50-footer exists for Curry in this dataset (his
    # actual max is 40ft).
    cases.append((
        "Curry, SYNTHETIC 50ft heave (distance overridden on the 40ft row)",
        curry_40_idx,
        {"SHOT_DISTANCE": 50, "distance_35_plus_ft": 1, "distance_28_35_ft": 0},
        "SYNTHETIC",
    ))

    # Case 2: Mason Plumlee (rotation big, not a star), real 5ft layup
    plumlee_5_idx = test_df[
        (test_df["PLAYER_NAME"] == "Mason Plumlee")
        & (test_df["SHOT_DISTANCE"] == 5)
        & (test_df["ACTION_TYPE"] == "Layup Shot")
    ].index[0]
    cases.append(("Mason Plumlee (rotation big), real 5ft layup [buzzer-beater, non-RA -- hard shot, see report]", plumlee_5_idx, None, "REAL"))

    # Case 2b: a CLEAN control -- same player, restricted-area, non-late-clock,
    # 0ft finish. Added after the above turned out to be an unusually hard
    # real shot (contested non-restricted-area buzzer-beater), not a
    # representative "easy layup".
    plumlee_clean_idx = test_df[
        (test_df["PLAYER_NAME"] == "Mason Plumlee")
        & (test_df["BASIC_ZONE"] == "Restricted Area")
        & (test_df["is_late_clock"] == 0)
        & (test_df["SHOT_DISTANCE"] <= 2)
    ].index[0]
    cases.append(("Mason Plumlee (rotation big), real 0-2ft restricted-area finish, non-late-clock [clean control]", plumlee_clean_idx, None, "REAL"))

    # Case 3: Curry, real 26ft pull-up 3 (made)
    curry_26_idx = test_df[
        (test_df["PLAYER_NAME"] == "Stephen Curry")
        & (test_df["SHOT_DISTANCE"] == 26)
        & (test_df["ACTION_TYPE"] == "Pullup Jump shot")
    ].index[0]
    cases.append(("Curry, real 26ft pull-up 3 (made)", curry_26_idx, None, "REAL"))

    # Case 4: Payton Pritchard (rotation guard, not a star), real 32ft deep 3 (missed)
    pritchard_32_idx = test_df[
        (test_df["PLAYER_NAME"] == "Payton Pritchard")
        & (test_df["SHOT_DISTANCE"] == 32)
    ].index[0]
    cases.append(("Payton Pritchard (rotation guard), real 32ft deep pull-up 3 (missed)", pritchard_32_idx, None, "REAL"))

    print("\n" + "=" * 115)
    print(f"{'Case':<65} {'OLD LogReg':>11} {'NEW LogReg':>11} {'OLD XGB':>9} {'NEW XGB':>9}")
    print("=" * 115)
    results = []
    for label, idx, overrides, kind in cases:
        old_lr = predict(old_logreg, OLD_NUMERIC, OLD_BINARY, idx, overrides)
        new_lr = predict(new_logreg, NEW_NUMERIC, NEW_BINARY, idx, overrides)
        old_xg = predict(old_xgb, OLD_NUMERIC, OLD_BINARY, idx, overrides)
        new_xg = predict(new_xgb_model, NEW_NUMERIC, NEW_BINARY, idx, overrides)
        row = test_df.loc[idx]
        print(f"{label:<65} {old_lr*100:>10.1f}% {new_lr*100:>10.1f}% {old_xg*100:>8.1f}% {new_xg*100:>8.1f}%")
        results.append({
            "case": label, "kind": kind,
            "shot_distance": float(overrides["SHOT_DISTANCE"]) if overrides else float(row["SHOT_DISTANCE"]),
            "actual_result": int(row["SHOT_MADE"]),
            "old_logreg_pred_pct": round(float(old_lr) * 100, 2),
            "new_logreg_pred_pct": round(float(new_lr) * 100, 2),
            "old_xgb_pred_pct": round(float(old_xg) * 100, 2),
            "new_xgb_pred_pct": round(float(new_xg) * 100, 2),
        })
    print("=" * 115)

    out_path = PROJECT_ROOT / "reports" / "edge_case_predictions.csv"
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
