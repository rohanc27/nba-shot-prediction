"""Compute leakage-free player historical shooting features.

For each shot, features are computed using only shots taken BEFORE
the current shot.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

PRIOR_WEIGHT = 100


def _smoothed_prior(prior_makes, prior_shots, prior_mean, prior_weight):
    return (prior_makes + prior_weight * prior_mean) / (
        prior_shots + prior_weight
    )


def _add_group_prior(
    df: pd.DataFrame,
    group_cols: list[str],
    prior_mean,
    prior_weight: int,
    pct_col: str,
    shots_col: str,
) -> pd.DataFrame:
    grouped = df.groupby(group_cols, sort=False)

    cum_makes_incl = grouped["SHOT_MADE"].cumsum()
    cum_shots_incl = grouped.cumcount() + 1

    prior_makes = cum_makes_incl - df["SHOT_MADE"]
    prior_shots = cum_shots_incl - 1

    df[pct_col] = _smoothed_prior(
        prior_makes=prior_makes,
        prior_shots=prior_shots,
        prior_mean=prior_mean,
        prior_weight=prior_weight,
    )

    df[shots_col] = prior_shots.astype(int)
    return df


def compute_player_priors(
    df: pd.DataFrame,
    prior_weight: int = PRIOR_WEIGHT,
) -> pd.DataFrame:
    df = df.copy()

    df["_within_game_order"] = (
        df["QUARTER"].astype(int) * 720
        + (12 - df["MINS_LEFT"].astype(int)) * 60
        + (60 - df["SECS_LEFT"].astype(int))
    )

    df = df.sort_values(
        ["PLAYER_ID", "GAME_DATE", "GAME_ID", "_within_game_order"],
        kind="stable",
    ).reset_index(drop=True)

    league_fg_pct = float(df["SHOT_MADE"].mean())
    print(f"  League prior FG%: {league_fg_pct:.4f}")
    print(f"  Prior weight: {prior_weight} shots")

    # Overall player FG%
    df = _add_group_prior(
        df=df,
        group_cols=["PLAYER_ID"],
        prior_mean=league_fg_pct,
        prior_weight=prior_weight,
        pct_col="player_prior_fg_pct",
        shots_col="player_prior_shots",
    )

    # Player 2PT / 3PT skill
    shot_type_means = df.groupby("SHOT_TYPE")["SHOT_MADE"].mean()
    df["_shot_type_prior"] = df["SHOT_TYPE"].map(shot_type_means)

    df = _add_group_prior(
        df=df,
        group_cols=["PLAYER_ID", "SHOT_TYPE"],
        prior_mean=df["_shot_type_prior"],
        prior_weight=prior_weight,
        pct_col="player_prior_shot_type_fg_pct",
        shots_col="player_prior_shot_type_shots",
    )

    df["player_prior_2pt_pct"] = df["player_prior_fg_pct"]
    df["player_prior_3pt_pct"] = df["player_prior_fg_pct"]
    df["player_prior_2pt_shots"] = 0
    df["player_prior_3pt_shots"] = 0

    is_2pt = df["SHOT_TYPE"] == "2PT Field Goal"
    is_3pt = df["SHOT_TYPE"] == "3PT Field Goal"

    df.loc[is_2pt, "player_prior_2pt_pct"] = df.loc[
        is_2pt, "player_prior_shot_type_fg_pct"
    ]
    df.loc[is_3pt, "player_prior_3pt_pct"] = df.loc[
        is_3pt, "player_prior_shot_type_fg_pct"
    ]

    df.loc[is_2pt, "player_prior_2pt_shots"] = df.loc[
        is_2pt, "player_prior_shot_type_shots"
    ]
    df.loc[is_3pt, "player_prior_3pt_shots"] = df.loc[
        is_3pt, "player_prior_shot_type_shots"
    ]

    # Player x zone skill
    zone_means = df.groupby("BASIC_ZONE")["SHOT_MADE"].mean()
    print(f"  Zone prior FG%s: {zone_means.to_dict()}")
    df["_zone_prior"] = df["BASIC_ZONE"].map(zone_means)

    df = _add_group_prior(
        df=df,
        group_cols=["PLAYER_ID", "BASIC_ZONE"],
        prior_mean=df["_zone_prior"],
        prior_weight=prior_weight,
        pct_col="player_prior_zone_fg_pct",
        shots_col="player_prior_zone_shots",
    )

    # Player x action skill
    action_means = df.groupby("action_category")["SHOT_MADE"].mean()
    print(f"  Action prior FG%s: {action_means.to_dict()}")
    df["_action_prior"] = df["action_category"].map(action_means)

    df = _add_group_prior(
        df=df,
        group_cols=["PLAYER_ID", "action_category"],
        prior_mean=df["_action_prior"],
        prior_weight=prior_weight,
        pct_col="player_prior_action_fg_pct",
        shots_col="player_prior_action_shots",
    )

    df = df.drop(
        columns=[
            "_within_game_order",
            "_shot_type_prior",
            "_zone_prior",
            "_action_prior",
            "player_prior_shot_type_fg_pct",
            "player_prior_shot_type_shots",
        ]
    )

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior-weight", type=int, default=PRIOR_WEIGHT)
    args = parser.parse_args()

    print("Loading raw shot data...")
    paths = sorted(RAW_DIR.glob("shots_*.parquet"))
    df = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    print(f"  Loaded {len(df):,} shots")

    df = df[df["BASIC_ZONE"] != "Backcourt"].copy()
    df = df.drop(
        columns=[c for c in ["POSITION", "POSITION_GROUP"] if c in df.columns]
    )

    if "action_category" not in df.columns:
        raise ValueError(
            "action_category missing. Run this through build_features.py."
        )

    print("\nComputing player priors...")
    df = compute_player_priors(df, prior_weight=args.prior_weight)

    out_path = PROCESSED_DIR / "shots_with_player_priors.parquet"
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)

    print(f"\nWritten to {out_path}")


if __name__ == "__main__":
    main()