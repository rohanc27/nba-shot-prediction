"""Build features from raw shot data for ML.

Reads parquet files from data/raw/, computes per-player priors,
engineers features, and writes a single combined parquet file to
data/processed/shots_features.parquet.

Run once after data pull:
    python -m src.features.build_features
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.build_player_features import compute_player_priors

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

ACTION_CATEGORY_MAP = {
    "Jump Shot": "jump_shot",
    "Pullup Jump shot": "pullup",
    "Step Back Jump shot": "stepback",
    "Fadeaway Jump Shot": "fadeaway",
    "Running Jump Shot": "running_jump",
    "Floating Jump shot": "floater",
    "Turnaround Jump Shot": "turnaround",
    "Hook Shot": "hook",
    "Running Hook Shot": "hook",
    "Driving Hook Shot": "hook",
    "Turnaround Hook Shot": "hook",
}

LAYUP_KEYWORDS = ("Layup", "Finger Roll", "Reverse Layup", "Cutting")
DUNK_KEYWORDS = ("Dunk", "Alley Oop", "Tip Dunk")
FLOATER_KEYWORDS = ("Floating", "Floater")


def categorize_action(action: str) -> str:
    if not isinstance(action, str):
        return "other"
    if any(k in action for k in DUNK_KEYWORDS):
        return "dunk"
    if any(k in action for k in LAYUP_KEYWORDS):
        return "layup"
    if any(k in action for k in FLOATER_KEYWORDS):
        return "floater"
    if action in ACTION_CATEGORY_MAP:
        return ACTION_CATEGORY_MAP[action]
    if "Pullup" in action or "Pull-Up" in action:
        return "pullup"
    if "Step Back" in action or "Stepback" in action:
        return "stepback"
    if "Fadeaway" in action:
        return "fadeaway"
    if "Hook" in action:
        return "hook"
    if "Jump Shot" in action or "Jumper" in action:
        return "jump_shot"
    return "other"


def compute_shot_angle(loc_x: pd.Series, loc_y: pd.Series) -> pd.Series:
    """Shot angle in degrees from straight-ahead (+Y axis)."""
    angle_rad = np.arctan2(loc_x, loc_y)
    return np.degrees(angle_rad)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer non-player features."""
    df = df.copy()

    df["shot_angle_deg"] = compute_shot_angle(df["LOC_X"], df["LOC_Y"])
    df["abs_angle_deg"] = df["shot_angle_deg"].abs()

    df["is_corner_3"] = df["BASIC_ZONE"].isin(
        ["Left Corner 3", "Right Corner 3"]
    ).astype(int)

    df["is_three"] = (df["SHOT_TYPE"] == "3PT Field Goal").astype(int)

    df["action_category"] = df["ACTION_TYPE"].apply(categorize_action)
    df["is_layup"] = (df["action_category"] == "layup").astype(int)
    df["is_dunk"] = (df["action_category"] == "dunk").astype(int)

    df["seconds_remaining_in_quarter"] = (
        df["MINS_LEFT"] * 60 + df["SECS_LEFT"]
    )
    df["is_late_clock"] = (df["seconds_remaining_in_quarter"] <= 24).astype(int)
    df["is_overtime"] = (df["QUARTER"] >= 5).astype(int)

    return df


def filter_shots(df: pd.DataFrame) -> pd.DataFrame:
    n_before = len(df)
    df = df[df["BASIC_ZONE"] != "Backcourt"].copy()
    print(f"  Dropped {n_before - len(df):,} backcourt shots")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROCESSED_DIR / "shots_features.parquet",
    )
    parser.add_argument("--prior-weight", type=int, default=100)
    args = parser.parse_args()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    paths = sorted(RAW_DIR.glob("shots_*.parquet"))
    print(f"Loading {len(paths)} raw files...")
    df = pd.concat(
        [pd.read_parquet(p) for p in paths], ignore_index=True
    )
    print(f"  Loaded {len(df):,} raw shots")

    drop_cols = ["POSITION", "POSITION_GROUP"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    print("Filtering...")
    df = filter_shots(df)

    print("\nEngineering non-player features...")
    df = build_features(df)

    print("\nComputing player priors (this takes ~30s)...")
    df = compute_player_priors(df, prior_weight=args.prior_weight)

    print(f"\nWriting {len(df):,} shots to {args.output}")
    df.to_parquet(args.output, index=False)

    print()
    print("=== Player prior summary ===")
    print(f"  player_prior_fg_pct: mean={df['player_prior_fg_pct'].mean():.4f}")
    print(f"  player_prior_zone_fg_pct: mean={df['player_prior_zone_fg_pct'].mean():.4f}")
    print(f"  player_prior_shots: median={df['player_prior_shots'].median():.0f}")

    print()
    print("=== action_category distribution ===")
    print(df["action_category"].value_counts())


if __name__ == "__main__":
    main()