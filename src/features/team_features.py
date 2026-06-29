"""Leakage-free historical team offensive shooting features."""

from __future__ import annotations

import pandas as pd

PRIOR_WEIGHT = 200


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


def compute_team_offense_priors(
    df: pd.DataFrame,
    prior_weight: int = PRIOR_WEIGHT,
) -> pd.DataFrame:
    """Compute leakage-free offensive team shooting features."""
    df = df.copy()

    df["_within_game_order"] = (
        df["QUARTER"].astype(int) * 720
        + (12 - df["MINS_LEFT"].astype(int)) * 60
        + (60 - df["SECS_LEFT"].astype(int))
    )

    df = df.sort_values(
        ["TEAM_ID", "GAME_DATE", "GAME_ID", "_within_game_order"],
        kind="stable",
    ).reset_index(drop=True)

    league_fg_pct = float(df["SHOT_MADE"].mean())
    print(f"  Team offense league prior FG%: {league_fg_pct:.4f}")

    # Team overall FG%
    df = _add_group_prior(
        df,
        ["TEAM_ID"],
        league_fg_pct,
        prior_weight,
        "team_prior_fg_pct",
        "team_prior_shots",
    )

    # Team x BASIC_ZONE
    zone_means = df.groupby("BASIC_ZONE")["SHOT_MADE"].mean()
    df["_zone_prior"] = df["BASIC_ZONE"].map(zone_means)

    df = _add_group_prior(
        df,
        ["TEAM_ID", "BASIC_ZONE"],
        df["_zone_prior"],
        prior_weight,
        "team_prior_zone_fg_pct",
        "team_prior_zone_shots",
    )

    # Team x SHOT_TYPE
    shot_type_means = df.groupby("SHOT_TYPE")["SHOT_MADE"].mean()
    df["_shot_type_prior"] = df["SHOT_TYPE"].map(shot_type_means)

    df = _add_group_prior(
        df,
        ["TEAM_ID", "SHOT_TYPE"],
        df["_shot_type_prior"],
        prior_weight,
        "team_prior_shot_type_fg_pct",
        "team_prior_shot_type_shots",
    )

    df["team_prior_2pt_pct"] = df["team_prior_fg_pct"]
    df["team_prior_3pt_pct"] = df["team_prior_fg_pct"]
    df["team_prior_2pt_shots"] = 0
    df["team_prior_3pt_shots"] = 0

    is_2pt = df["SHOT_TYPE"] == "2PT Field Goal"
    is_3pt = df["SHOT_TYPE"] == "3PT Field Goal"

    df.loc[is_2pt, "team_prior_2pt_pct"] = df.loc[
        is_2pt, "team_prior_shot_type_fg_pct"
    ]
    df.loc[is_3pt, "team_prior_3pt_pct"] = df.loc[
        is_3pt, "team_prior_shot_type_fg_pct"
    ]

    df.loc[is_2pt, "team_prior_2pt_shots"] = df.loc[
        is_2pt, "team_prior_shot_type_shots"
    ]
    df.loc[is_3pt, "team_prior_3pt_shots"] = df.loc[
        is_3pt, "team_prior_shot_type_shots"
    ]

    df = df.drop(
        columns=[
            "_within_game_order",
            "_zone_prior",
            "_shot_type_prior",
            "team_prior_shot_type_fg_pct",
            "team_prior_shot_type_shots",
        ]
    )

    return df
