"""Leakage-free opponent defensive shooting features."""

from __future__ import annotations

import pandas as pd

PRIOR_WEIGHT = 200


def _smoothed_prior(prior_makes_allowed, prior_shots_allowed, prior_mean, prior_weight):
    return (prior_makes_allowed + prior_weight * prior_mean) / (
        prior_shots_allowed + prior_weight
    )


def _add_allowed_group_prior(
    df: pd.DataFrame,
    group_cols: list[str],
    prior_mean,
    prior_weight: int,
    pct_col: str,
    shots_col: str,
) -> pd.DataFrame:
    grouped = df.groupby(group_cols, sort=False)

    cum_makes_allowed_incl = grouped["SHOT_MADE"].cumsum()
    cum_shots_allowed_incl = grouped.cumcount() + 1

    prior_makes_allowed = cum_makes_allowed_incl - df["SHOT_MADE"]
    prior_shots_allowed = cum_shots_allowed_incl - 1

    df[pct_col] = _smoothed_prior(
        prior_makes_allowed=prior_makes_allowed,
        prior_shots_allowed=prior_shots_allowed,
        prior_mean=prior_mean,
        prior_weight=prior_weight,
    )
    df[shots_col] = prior_shots_allowed.astype(int)

    return df


def add_opponent_column(df: pd.DataFrame) -> pd.DataFrame:
    """Infer opponent from shooting team and home/away teams."""
    df = df.copy()

    df["OPPONENT_TEAM"] = df["AWAY_TEAM"]
    is_away_shooter = df["TEAM_NAME"] == df["AWAY_TEAM"]
    df.loc[is_away_shooter, "OPPONENT_TEAM"] = df.loc[
        is_away_shooter, "HOME_TEAM"
    ]

    df["is_home"] = (df["TEAM_NAME"] == df["HOME_TEAM"]).astype(int)

    return df


def compute_opponent_defense_priors(
    df: pd.DataFrame,
    prior_weight: int = PRIOR_WEIGHT,
) -> pd.DataFrame:
    """Compute leakage-free opponent defensive allowed-FG features."""
    df = add_opponent_column(df)

    df["_within_game_order"] = (
        df["QUARTER"].astype(int) * 720
        + (12 - df["MINS_LEFT"].astype(int)) * 60
        + (60 - df["SECS_LEFT"].astype(int))
    )

    df = df.sort_values(
        ["OPPONENT_TEAM", "GAME_DATE", "GAME_ID", "_within_game_order"],
        kind="stable",
    ).reset_index(drop=True)

    league_fg_pct = float(df["SHOT_MADE"].mean())
    print(f"  Opponent defense league prior allowed FG%: {league_fg_pct:.4f}")

    # Opponent overall allowed FG%
    df = _add_allowed_group_prior(
        df,
        ["OPPONENT_TEAM"],
        league_fg_pct,
        prior_weight,
        "opponent_allowed_fg_pct",
        "opponent_allowed_shots",
    )

    # Opponent x BASIC_ZONE allowed FG%
    zone_means = df.groupby("BASIC_ZONE")["SHOT_MADE"].mean()
    df["_zone_prior"] = df["BASIC_ZONE"].map(zone_means)

    df = _add_allowed_group_prior(
        df,
        ["OPPONENT_TEAM", "BASIC_ZONE"],
        df["_zone_prior"],
        prior_weight,
        "opponent_allowed_zone_fg_pct",
        "opponent_allowed_zone_shots",
    )

    # Opponent x SHOT_TYPE allowed FG%
    shot_type_means = df.groupby("SHOT_TYPE")["SHOT_MADE"].mean()
    df["_shot_type_prior"] = df["SHOT_TYPE"].map(shot_type_means)

    df = _add_allowed_group_prior(
        df,
        ["OPPONENT_TEAM", "SHOT_TYPE"],
        df["_shot_type_prior"],
        prior_weight,
        "opponent_allowed_shot_type_fg_pct",
        "opponent_allowed_shot_type_shots",
    )

    df["opponent_allowed_2pt_pct"] = df["opponent_allowed_fg_pct"]
    df["opponent_allowed_3pt_pct"] = df["opponent_allowed_fg_pct"]
    df["opponent_allowed_2pt_shots"] = 0
    df["opponent_allowed_3pt_shots"] = 0

    is_2pt = df["SHOT_TYPE"] == "2PT Field Goal"
    is_3pt = df["SHOT_TYPE"] == "3PT Field Goal"

    df.loc[is_2pt, "opponent_allowed_2pt_pct"] = df.loc[
        is_2pt, "opponent_allowed_shot_type_fg_pct"
    ]
    df.loc[is_3pt, "opponent_allowed_3pt_pct"] = df.loc[
        is_3pt, "opponent_allowed_shot_type_fg_pct"
    ]

    df.loc[is_2pt, "opponent_allowed_2pt_shots"] = df.loc[
        is_2pt, "opponent_allowed_shot_type_shots"
    ]
    df.loc[is_3pt, "opponent_allowed_3pt_shots"] = df.loc[
        is_3pt, "opponent_allowed_shot_type_shots"
    ]

    df = df.drop(
        columns=[
            "_within_game_order",
            "_zone_prior",
            "_shot_type_prior",
            "opponent_allowed_shot_type_fg_pct",
            "opponent_allowed_shot_type_shots",
        ]
    )

    return df
