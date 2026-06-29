"""Leakage-free player shot tendency features.

These features measure how often a player historically takes each kind
of shot before the current attempt.
"""

from __future__ import annotations

import pandas as pd

PRIOR_WEIGHT = 100


def _smoothed_rate(prior_count, prior_total, league_rate, prior_weight):
    return (prior_count + prior_weight * league_rate) / (
        prior_total + prior_weight
    )


def _add_tendency(
    df: pd.DataFrame,
    group_col: str,
    output_col: str,
    prior_weight: int,
) -> pd.DataFrame:
    league_rates = df[group_col].value_counts(normalize=True)
    df[f"_{group_col}_league_rate"] = df[group_col].map(league_rates)

    grouped_player = df.groupby("PLAYER_ID", sort=False)
    prior_total = grouped_player.cumcount()

    grouped_player_type = df.groupby(["PLAYER_ID", group_col], sort=False)
    prior_type_count = grouped_player_type.cumcount()

    df[output_col] = _smoothed_rate(
        prior_count=prior_type_count,
        prior_total=prior_total,
        league_rate=df[f"_{group_col}_league_rate"],
        prior_weight=prior_weight,
    )

    df = df.drop(columns=[f"_{group_col}_league_rate"])
    return df


def compute_player_tendencies(
    df: pd.DataFrame,
    prior_weight: int = PRIOR_WEIGHT,
) -> pd.DataFrame:
    """Compute player historical shot mix features."""
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

    df = _add_tendency(
        df,
        group_col="BASIC_ZONE",
        output_col="player_tendency_zone_rate",
        prior_weight=prior_weight,
    )

    df = _add_tendency(
        df,
        group_col="action_category",
        output_col="player_tendency_action_rate",
        prior_weight=prior_weight,
    )

    df = _add_tendency(
        df,
        group_col="shot_profile",
        output_col="player_tendency_shot_profile_rate",
        prior_weight=prior_weight,
    )

    df = df.drop(columns=["_within_game_order"])
    return df
