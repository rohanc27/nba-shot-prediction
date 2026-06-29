"""Player residual shooting skill features.

Measures how much better (or worse) a player shoots than league average
for different shot contexts using only prior shots.
"""

from __future__ import annotations

import pandas as pd

PRIOR_WEIGHT = 100


def _compute_residual(
    df: pd.DataFrame,
    group_col: str,
    output_col: str,
):
    league_means = df.groupby(group_col)["SHOT_MADE"].mean()

    league_lookup = df[group_col].map(league_means)

    grouped = df.groupby(["PLAYER_ID", group_col], sort=False)

    cum_makes = grouped["SHOT_MADE"].cumsum()
    cum_shots = grouped.cumcount() + 1

    prior_makes = cum_makes - df["SHOT_MADE"]
    prior_shots = cum_shots - 1

    player_rate = (
        prior_makes + PRIOR_WEIGHT * league_lookup
    ) / (
        prior_shots + PRIOR_WEIGHT
    )

    df[output_col] = player_rate - league_lookup

    return df


def compute_player_residual_features(df):

    df = df.copy()

    df = _compute_residual(
        df,
        "BASIC_ZONE",
        "player_zone_residual",
    )

    df = _compute_residual(
        df,
        "action_category",
        "player_action_residual",
    )

    df = _compute_residual(
        df,
        "shot_profile",
        "player_profile_residual",
    )

    return df
