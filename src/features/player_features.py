"""Leakage-free historical player shooting features."""

from __future__ import annotations

import pandas as pd

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
    """Compute all player-history features without target leakage."""
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
        df,
        ["PLAYER_ID"],
        league_fg_pct,
        prior_weight,
        "player_prior_fg_pct",
        "player_prior_shots",
    )

    # Player x shot type: 2PT / 3PT
    shot_type_means = df.groupby("SHOT_TYPE")["SHOT_MADE"].mean()
    df["_shot_type_prior"] = df["SHOT_TYPE"].map(shot_type_means)

    df = _add_group_prior(
        df,
        ["PLAYER_ID", "SHOT_TYPE"],
        df["_shot_type_prior"],
        prior_weight,
        "player_prior_shot_type_fg_pct",
        "player_prior_shot_type_shots",
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

    # Player x BASIC_ZONE
    zone_means = df.groupby("BASIC_ZONE")["SHOT_MADE"].mean()
    print(f"  BASIC_ZONE prior FG%s: {zone_means.to_dict()}")
    df["_zone_prior"] = df["BASIC_ZONE"].map(zone_means)

    df = _add_group_prior(
        df,
        ["PLAYER_ID", "BASIC_ZONE"],
        df["_zone_prior"],
        prior_weight,
        "player_prior_zone_fg_pct",
        "player_prior_zone_shots",
    )

    # Player x ZONE_NAME
    zone_name_means = df.groupby("ZONE_NAME")["SHOT_MADE"].mean()
    print(f"  ZONE_NAME prior count: {len(zone_name_means)}")
    df["_zone_name_prior"] = df["ZONE_NAME"].map(zone_name_means)

    df = _add_group_prior(
        df,
        ["PLAYER_ID", "ZONE_NAME"],
        df["_zone_name_prior"],
        prior_weight,
        "player_prior_zone_name_fg_pct",
        "player_prior_zone_name_shots",
    )

    # Player x ZONE_RANGE
    zone_range_means = df.groupby("ZONE_RANGE")["SHOT_MADE"].mean()
    print(f"  ZONE_RANGE prior count: {len(zone_range_means)}")
    df["_zone_range_prior"] = df["ZONE_RANGE"].map(zone_range_means)

    df = _add_group_prior(
        df,
        ["PLAYER_ID", "ZONE_RANGE"],
        df["_zone_range_prior"],
        prior_weight,
        "player_prior_zone_range_fg_pct",
        "player_prior_zone_range_shots",
    )

    # Player x action category
    action_means = df.groupby("action_category")["SHOT_MADE"].mean()
    print(f"  Action prior FG%s: {action_means.to_dict()}")
    df["_action_prior"] = df["action_category"].map(action_means)

    df = _add_group_prior(
        df,
        ["PLAYER_ID", "action_category"],
        df["_action_prior"],
        prior_weight,
        "player_prior_action_fg_pct",
        "player_prior_action_shots",
    )

    df = df.drop(
        columns=[
            "_within_game_order",
            "_shot_type_prior",
            "_zone_prior",
            "_zone_name_prior",
            "_zone_range_prior",
            "_action_prior",
            "player_prior_shot_type_fg_pct",
            "player_prior_shot_type_shots",
        ]
    )

    return df
