"""Distance-stratified player priors -- leakage-free, like player_features.py.

player_prior_fg_pct (in player_features.py) is a single number aggregated
across every distance a player has ever shot from. That's the mechanism
behind the diagnosed bug: Steph Curry's ~45% aggregate FG% gets applied
uniformly whether he's finishing at the rim or heaving a 50-footer, and
nothing in the feature set tells the model his accuracy actually collapses
at extreme range. This module computes his expanding-window FG% *within
each distance bucket separately*, so a 50-foot shot sees his history on
50-foot shots (a handful of makes out of dozens of attempts), not his
history everywhere.

Must run after compute_player_priors() (needs player_prior_fg_pct/
player_prior_shots as the fallback for buckets a player hasn't shot from
yet) and after build_features() (needs the distance_*_ft columns' input,
SHOT_DISTANCE, to already be present -- though this module derives its
own bucket labels rather than reading the binary columns directly, so it
only actually needs SHOT_DISTANCE).
"""
from __future__ import annotations

import pandas as pd

from src.features.player_features import _add_group_prior

BUCKET_EDGES = [-0.01, 10, 16, 24, 28, 35, float("inf")]
BUCKET_LABELS = ["0_10", "10_16", "16_24", "24_28", "28_35", "35_plus"]


def compute_distance_stratified_player_priors(
    df: pd.DataFrame,
    prior_weight: int = 100,
) -> pd.DataFrame:
    """Add player_prior_fg_pct_<bucket> / player_prior_shots_<bucket> for
    each of the 6 distance buckets.

    For a shot in bucket X, player_prior_fg_pct_X holds the player's
    real expanding-window FG% on prior shots in bucket X (Bayesian-
    smoothed toward that bucket's league-average FG%, same mechanism as
    player_features.py). For buckets the shot is NOT in, the column falls
    back to the player's overall player_prior_fg_pct -- mirroring the
    existing player_prior_2pt_pct/player_prior_3pt_pct pattern in
    player_features.py, rather than leaving it at 0/NaN, so the column is
    always a sensible number even when it isn't this shot's own bucket.
    """
    required = {"player_prior_fg_pct", "player_prior_shots"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"compute_distance_stratified_player_priors requires {missing} "
            "-- run compute_player_priors() first."
        )

    df = df.copy()

    # Re-derive the same temporal order player_features.py uses. Safe to
    # redo even if the df arrives pre-sorted (idempotent), and makes this
    # function correct regardless of call order in the pipeline.
    df["_within_game_order"] = (
        df["QUARTER"].astype(int) * 720
        + (12 - df["MINS_LEFT"].astype(int)) * 60
        + (60 - df["SECS_LEFT"].astype(int))
    )
    df = df.sort_values(
        ["PLAYER_ID", "GAME_DATE", "GAME_ID", "_within_game_order"],
        kind="stable",
    ).reset_index(drop=True)

    df["_distance_bucket"] = pd.cut(
        df["SHOT_DISTANCE"], bins=BUCKET_EDGES, labels=BUCKET_LABELS,
    )

    bucket_means = df.groupby("_distance_bucket", observed=True)["SHOT_MADE"].mean()
    print(f"  Distance-bucket league FG%s: {bucket_means.to_dict()}")
    df["_distance_bucket_prior"] = df["_distance_bucket"].map(bucket_means).astype(float)

    df = _add_group_prior(
        df,
        ["PLAYER_ID", "_distance_bucket"],
        df["_distance_bucket_prior"],
        prior_weight,
        "_bucket_fg_pct",
        "_bucket_shots",
    )

    for label in BUCKET_LABELS:
        pct_col = f"player_prior_fg_pct_{label}"
        shots_col = f"player_prior_shots_{label}"
        in_bucket = df["_distance_bucket"] == label

        df[pct_col] = df["player_prior_fg_pct"]
        df[shots_col] = 0
        df.loc[in_bucket, pct_col] = df.loc[in_bucket, "_bucket_fg_pct"]
        df.loc[in_bucket, shots_col] = df.loc[in_bucket, "_bucket_shots"]
        df[shots_col] = df[shots_col].astype(int)

    df = df.drop(
        columns=[
            "_within_game_order",
            "_distance_bucket",
            "_distance_bucket_prior",
            "_bucket_fg_pct",
            "_bucket_shots",
        ]
    )

    return df
