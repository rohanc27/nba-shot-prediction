"""Build small app lookup table for Streamlit deployment.

The app does not need the full 654k-shot dataset. It only needs one
representative/latest row per player plus a few league-average context
tables.

Usage:
    python -m src.models.build_app_lookup
"""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

FEATURES_FOR_APP = [
    "PLAYER_NAME",
    "GAME_DATE",
    "GAME_ID",
    "QUARTER",
    "MINS_LEFT",
    "SECS_LEFT",
    "SHOT_DISTANCE",
    "LOC_X",
    "LOC_Y",
    "shot_angle_deg",
    "abs_angle_deg",
    "seconds_remaining_in_quarter",
    "player_prior_fg_pct",
    "player_prior_zone_fg_pct",
    "player_prior_zone_name_fg_pct",
    "player_prior_zone_range_fg_pct",
    "player_prior_action_fg_pct",
    "player_prior_2pt_pct",
    "player_prior_3pt_pct",
    "player_prior_shots",
    "player_prior_zone_shots",
    "player_prior_zone_name_shots",
    "player_prior_zone_range_shots",
    "player_prior_action_shots",
    "player_prior_2pt_shots",
    "player_prior_3pt_shots",
    "team_prior_fg_pct",
    "team_prior_zone_fg_pct",
    "team_prior_2pt_pct",
    "team_prior_3pt_pct",
    "team_prior_shots",
    "team_prior_zone_shots",
    "team_prior_2pt_shots",
    "team_prior_3pt_shots",
    "opponent_allowed_fg_pct",
    "opponent_allowed_zone_fg_pct",
    "opponent_allowed_2pt_pct",
    "opponent_allowed_3pt_pct",
    "opponent_allowed_shots",
    "opponent_allowed_zone_shots",
    "opponent_allowed_2pt_shots",
    "opponent_allowed_3pt_shots",
    "player_tendency_zone_rate",
    "player_tendency_action_rate",
    "player_tendency_shot_profile_rate",
    "player_zone_residual",
    "player_action_residual",
    "player_profile_residual",
    "is_three",
    "is_corner_3",
    "is_layup",
    "is_dunk",
    "is_late_clock",
    "is_overtime",
    "is_home",
    "action_category",
    "BASIC_ZONE",
    "ZONE_NAME",
    "ZONE_RANGE",
    "shot_profile",
    "zone_range",
    "SHOT_MADE",
]


def main() -> None:
    df = pd.read_parquet(PROCESSED_DIR / "shots_features.parquet")

    df = df.sort_values(
        ["PLAYER_NAME", "GAME_DATE", "GAME_ID", "QUARTER", "MINS_LEFT", "SECS_LEFT"]
    )

    latest_player_rows = (
        df.groupby("PLAYER_NAME", as_index=False)
        .tail(1)[FEATURES_FOR_APP]
        .reset_index(drop=True)
    )

    player_summary = df.groupby("PLAYER_NAME").agg(
        app_shots=("SHOT_MADE", "count"),
        app_fg_pct=("SHOT_MADE", "mean"),
    ).reset_index()

    three_summary = (
        df[df["is_three"] == 1]
        .groupby("PLAYER_NAME")
        .agg(app_three_pct=("SHOT_MADE", "mean"))
        .reset_index()
    )

    rim_summary = (
        df[df["BASIC_ZONE"] == "Restricted Area"]
        .groupby("PLAYER_NAME")
        .agg(app_rim_pct=("SHOT_MADE", "mean"))
        .reset_index()
    )

    lookup = latest_player_rows.merge(player_summary, on="PLAYER_NAME", how="left")
    lookup = lookup.merge(three_summary, on="PLAYER_NAME", how="left")
    lookup = lookup.merge(rim_summary, on="PLAYER_NAME", how="left")

    league_context = df.groupby(["BASIC_ZONE", "action_category"]).agg(
        league_context_fg_pct=("SHOT_MADE", "mean"),
        league_context_shots=("SHOT_MADE", "count"),
    ).reset_index()

    league_zone = df.groupby("BASIC_ZONE").agg(
        league_zone_fg_pct=("SHOT_MADE", "mean"),
        league_zone_shots=("SHOT_MADE", "count"),
    ).reset_index()

    out_lookup = PROCESSED_DIR / "app_player_lookup.parquet"
    out_context = PROCESSED_DIR / "app_league_context.parquet"
    out_zone = PROCESSED_DIR / "app_league_zone.parquet"

    lookup.to_parquet(out_lookup, index=False)
    league_context.to_parquet(out_context, index=False)
    league_zone.to_parquet(out_zone, index=False)

    print(f"Saved {out_lookup} ({len(lookup):,} players)")
    print(f"Saved {out_context} ({len(league_context):,} contexts)")
    print(f"Saved {out_zone} ({len(league_zone):,} zones)")


if __name__ == "__main__":
    main()
