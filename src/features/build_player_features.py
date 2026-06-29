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

    For each shot, the player_prior_* features describe what we know
    about the player from shots taken BEFORE this one. The current
    shot's outcome is never included in its own prior.

    Computed features:
        - player_prior_fg_pct: overall FG% before this shot
        - player_prior_zone_fg_pct: FG% in this zone before this shot
        - player_prior_shots: total prior attempts (sample size)
        - player_prior_zone_shots: prior attempts in this zone
    """
    df = df.copy()

    # Sort by player, then chronologically by game and within-game time.
    # Note: within-game ordering by game time isn't perfectly available
    # from this dataset (no event order), but per-game ordering is correct.
    # We use GAME_DATE + GAME_ID + QUARTER + (12 - MINS_LEFT) + (60 - SECS_LEFT)
    # as a within-game proxy.
    df["_within_game_order"] = (
        df["QUARTER"].astype(int) * 720
        + (12 - df["MINS_LEFT"].astype(int)) * 60
        + (60 - df["SECS_LEFT"].astype(int))
    )

    df = df.sort_values(
        ["PLAYER_ID", "GAME_DATE", "GAME_ID", "_within_game_order"],
        kind="stable",
    ).reset_index(drop=True)

    # League average FG% as the Bayesian prior mean
    league_fg_pct = float(df["SHOT_MADE"].mean())
    print(f"  League prior FG%: {league_fg_pct:.4f}")
    print(f"  Prior weight: {prior_weight} shots")

    # ---- Per-player overall expanding FG% ----
    grouped = df.groupby("PLAYER_ID", sort=False)
    # shift(1) so the current shot is NOT included in its own prior
    cumulative_makes = grouped["SHOT_MADE"].cumsum().shift(1)
    cumulative_shots = grouped["SHOT_MADE"].cumcount()

    # First shot of each player gets 0 prior makes, 0 prior shots
    # — we need to zero those out where shift produced NaN.
    cumulative_makes = cumulative_makes.fillna(0)

    # Reset cumulative_makes to zero at each player boundary
    # (shift(1) within group via cumsum is correct, but the .shift(1)
    # across the dataframe leaks across players. We need to mask.)
    player_boundary = df["PLAYER_ID"] != df["PLAYER_ID"].shift(1)
    # On boundary rows, cumulative_makes should be 0 (no prior shots)
    cumulative_makes = np.where(player_boundary, 0, cumulative_makes)
    cumulative_makes = pd.Series(cumulative_makes, index=df.index)

    # Bayesian-smoothed FG%
    df["player_prior_fg_pct"] = (
        (cumulative_makes + prior_weight * league_fg_pct)
        / (cumulative_shots + prior_weight)
    )
    df["player_prior_shots"] = cumulative_shots.astype(int)

    # ---- Per-player-per-zone expanding FG% ----
    # League prior FG% by zone (to use as the prior mean for that zone)
    zone_means = df.groupby("BASIC_ZONE")["SHOT_MADE"].mean()
    print(f"  Zone prior FG%s: {zone_means.to_dict()}")
    df["_zone_league_fg_pct"] = df["BASIC_ZONE"].map(zone_means)

    # Group by (player, zone) and compute rolling sums
    pz_grouped = df.groupby(["PLAYER_ID", "BASIC_ZONE"], sort=False)
    pz_cum_makes = pz_grouped["SHOT_MADE"].cumsum().shift(1).fillna(0)
    pz_cum_shots = pz_grouped["SHOT_MADE"].cumcount()

    pz_boundary = (
        (df["PLAYER_ID"] != df["PLAYER_ID"].shift(1))
        | (df["BASIC_ZONE"] != df["BASIC_ZONE"].shift(1))
    )
    pz_cum_makes = np.where(pz_boundary, 0, pz_cum_makes)
    pz_cum_makes = pd.Series(pz_cum_makes, index=df.index)

    # Bayesian smoothing using zone-specific prior
    df["player_prior_zone_fg_pct"] = (
        (pz_cum_makes + prior_weight * df["_zone_league_fg_pct"])
        / (pz_cum_shots + prior_weight)
    )
    df["player_prior_zone_shots"] = pz_cum_shots.astype(int)

    # Clean up temporary columns
    df = df.drop(columns=["_within_game_order", "_zone_league_fg_pct"])

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
