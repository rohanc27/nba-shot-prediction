"""Build features from raw shot data for ML.

Reads parquet files from data/raw/, engineers features, and writes
a single combined parquet file to data/processed/shots_features.parquet.

Run once after data pull:
    python -m src.features.build_features
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Map raw ACTION_TYPE strings to coarser categories
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

# Layups and dunks need their own categories — much higher make rates
LAYUP_KEYWORDS = ("Layup", "Finger Roll", "Reverse Layup", "Cutting")
DUNK_KEYWORDS = ("Dunk", "Alley Oop", "Tip Dunk")
DRIVING_KEYWORDS = ("Driving",)
FLOATER_KEYWORDS = ("Floating", "Floater")


def categorize_action(action: str) -> str:
    """Map a raw ACTION_TYPE string to a coarse category."""
    if not isinstance(action, str):
        return "other"

    # Order matters — dunks before layups, layups before generic
    if any(k in action for k in DUNK_KEYWORDS):
        return "dunk"
    if any(k in action for k in LAYUP_KEYWORDS):
        return "layup"
    if any(k in action for k in FLOATER_KEYWORDS):
        return "floater"

    # Check explicit mapping
    if action in ACTION_CATEGORY_MAP:
        return ACTION_CATEGORY_MAP[action]

    # Fallbacks based on keywords
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
    """Compute shot angle in degrees from the basket.

    The basket is at (0, 0). Angle is measured from the +Y axis
    (straight-ahead = 0), with negative values to the left (negative X)
    and positive to the right.

    Returns angle in degrees, range approximately [-90, +90].
    A shot at the right corner is +90, left corner -90, top of the
    key is 0.
    """
    # arctan2(x, y) gives angle from +Y axis when args are (x, y)
    # rather than the usual (y, x). Negative X -> negative angle.
    angle_rad = np.arctan2(loc_x, loc_y)
    return np.degrees(angle_rad)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer features from raw shot data."""
    df = df.copy()

    # --- Location features ---
    df["shot_angle_deg"] = compute_shot_angle(df["LOC_X"], df["LOC_Y"])
    df["abs_angle_deg"] = df["shot_angle_deg"].abs()

    # Corner 3s vs above-the-break 3s
    df["is_corner_3"] = df["BASIC_ZONE"].isin(
        ["Left Corner 3", "Right Corner 3"]
    ).astype(int)

    # --- Shot type ---
    df["is_three"] = (df["SHOT_TYPE"] == "3PT Field Goal").astype(int)

    # --- Action category ---
    df["action_category"] = df["ACTION_TYPE"].apply(categorize_action)
    df["is_layup"] = (df["action_category"] == "layup").astype(int)
    df["is_dunk"] = (df["action_category"] == "dunk").astype(int)

    # --- Game context ---
    df["seconds_remaining_in_quarter"] = (
        df["MINS_LEFT"] * 60 + df["SECS_LEFT"]
    )
    df["is_late_clock"] = (df["seconds_remaining_in_quarter"] <= 24).astype(int)
    df["is_overtime"] = (df["QUARTER"] >= 5).astype(int)

    return df


def filter_shots(df: pd.DataFrame) -> pd.DataFrame:
    """Filter out shots we don't want to model."""
    n_before = len(df)

    # Drop backcourt heaves — not real shooting attempts
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
    args = parser.parse_args()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    paths = sorted(RAW_DIR.glob("shots_*.parquet"))
    print(f"Loading {len(paths)} raw files...")
    df = pd.concat(
        [pd.read_parquet(p) for p in paths], ignore_index=True
    )
    print(f"  Loaded {len(df):,} raw shots")

    # Drop columns missing from any season (POSITION, POSITION_GROUP missing in 2024-25)
    drop_cols = ["POSITION", "POSITION_GROUP"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    print("Filtering...")
    df = filter_shots(df)

    print("Engineering features...")
    df = build_features(df)

    print(f"Writing {len(df):,} shots to {args.output}")
    df.to_parquet(args.output, index=False)

    # Print a feature summary
    print()
    print("=== Feature summary ===")
    new_features = [
        "shot_angle_deg", "abs_angle_deg", "is_corner_3",
        "is_three", "action_category", "is_layup", "is_dunk",
        "seconds_remaining_in_quarter", "is_late_clock", "is_overtime",
    ]
    for f in new_features:
        if df[f].dtype in ("int64", "int32", "float64", "float32"):
            print(f"  {f}: mean={df[f].mean():.3f}, "
                  f"range=[{df[f].min():.2f}, {df[f].max():.2f}]")
        else:
            print(f"  {f}: {df[f].nunique()} unique values")

    print()
    print("=== action_category distribution ===")
    print(df["action_category"].value_counts())


if __name__ == "__main__":
    main()
