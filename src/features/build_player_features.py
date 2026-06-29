"""Compute per-player, per-zone expanding-window FG% with Bayesian smoothing.

For each shot in the dataset, computes the player's prior performance
using only shots taken BEFORE the current shot — strictly causal, no
leakage. Smoothed against the league average with a configurable
prior weight.

Run as part of the feature pipeline; called by build_features.py.

Usage (standalone):
    python -m src.features.build_player_features
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Bayesian prior weight: equivalent to N shots at league average.
# Higher = stronger pull toward the league mean for low-sample players.
# 100 is reasonable for shot data (about 30 games of attempts).
PRIOR_WEIGHT = 100


def compute_player_priors(
    df: pd.DataFrame,
    prior_weight: int = PRIOR_WEIGHT,
) -> pd.DataFrame:
    """Add per-player rolling FG% features without leakage.

    For each shot, the player_prior_* features describe the player's
    history of shots taken BEFORE this one. The current shot's outcome
    is never included in its own prior.

    Implementation: we compute the inclusive cumulative sum within each
    group, then subtract the current row's value to get a strict prior.
    This is mathematically equivalent to "shift down by one within group"
    but works correctly when groups are non-contiguous in the dataframe
    (which is the case for player x zone groups, since players' shots
    from different zones are interleaved chronologically).
    """
    df = df.copy()

    # Within-game ordering for consistent chronological sort
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

    # --- Per-player overall priors ---
    grouped = df.groupby("PLAYER_ID", sort=False)
    cum_makes_incl = grouped["SHOT_MADE"].cumsum()           # includes current
    cum_shots_incl = grouped.cumcount() + 1                  # 1-indexed
    # Strict prior: exclude the current row
    prior_makes = cum_makes_incl - df["SHOT_MADE"]
    prior_shots = cum_shots_incl - 1                         # = cumcount()

    df["player_prior_fg_pct"] = (
        (prior_makes + prior_weight * league_fg_pct)
        / (prior_shots + prior_weight)
    )
    df["player_prior_shots"] = prior_shots.astype(int)

    # --- Per-player-per-zone priors ---
    zone_means = df.groupby("BASIC_ZONE")["SHOT_MADE"].mean()
    print(f"  Zone prior FG%s: {zone_means.to_dict()}")
    zone_league_fg_pct = df["BASIC_ZONE"].map(zone_means)

    pz_grouped = df.groupby(["PLAYER_ID", "BASIC_ZONE"], sort=False)
    pz_cum_makes_incl = pz_grouped["SHOT_MADE"].cumsum()
    pz_cum_shots_incl = pz_grouped.cumcount() + 1
    pz_prior_makes = pz_cum_makes_incl - df["SHOT_MADE"]
    pz_prior_shots = pz_cum_shots_incl - 1

    df["player_prior_zone_fg_pct"] = (
        (pz_prior_makes + prior_weight * zone_league_fg_pct)
        / (pz_prior_shots + prior_weight)
    )
    df["player_prior_zone_shots"] = pz_prior_shots.astype(int)

    df = df.drop(columns=["_within_game_order"])

    return df

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prior-weight",
        type=int,
        default=PRIOR_WEIGHT,
    )
    args = parser.parse_args()

    print("Loading raw shot data...")
    paths = sorted(RAW_DIR.glob("shots_*.parquet"))
    df = pd.concat(
        [pd.read_parquet(p) for p in paths], ignore_index=True
    )
    print(f"  Loaded {len(df):,} shots")

    # Drop backcourt (consistent with build_features.py)
    df = df[df["BASIC_ZONE"] != "Backcourt"].copy()

    # Drop columns missing from some seasons
    df = df.drop(
        columns=[c for c in ["POSITION", "POSITION_GROUP"] if c in df.columns]
    )

    print("\nComputing player priors...")
    df = compute_player_priors(df, prior_weight=args.prior_weight)

    print("\n=== Player prior feature summary ===")
    print(f"  player_prior_fg_pct: "
          f"mean={df['player_prior_fg_pct'].mean():.4f}, "
          f"std={df['player_prior_fg_pct'].std():.4f}")
    print(f"  player_prior_zone_fg_pct: "
          f"mean={df['player_prior_zone_fg_pct'].mean():.4f}, "
          f"std={df['player_prior_zone_fg_pct'].std():.4f}")
    print(f"  player_prior_shots: "
          f"mean={df['player_prior_shots'].mean():.1f}, "
          f"median={df['player_prior_shots'].median():.0f}, "
          f"max={df['player_prior_shots'].max()}")
    print(f"  Shots with zero prior history: "
          f"{(df['player_prior_shots'] == 0).sum():,} "
          f"({100 * (df['player_prior_shots'] == 0).mean():.2f}%)")

    out_path = PROCESSED_DIR / "shots_with_player_priors.parquet"
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"\nWritten to {out_path}")


if __name__ == "__main__":
    main()
