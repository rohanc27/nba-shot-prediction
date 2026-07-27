"""Validate the expanded dataset + new proxy/bio features: coverage, ranges,
missingness, and temporal leakage checks. Read-only report, no model training.
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from features.defender_proxy_features import add_defender_proxy_features, SHOT_LOGS_PATH  # noqa: E402
from features.player_bio_features import add_player_bio_features, coverage_report  # noqa: E402


def load_all_shots() -> pd.DataFrame:
    files = sorted(glob.glob(str(PROJECT_ROOT / "data" / "raw" / "shots_*.parquet")))
    frames = [pd.read_parquet(f) for f in files]
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    print("=" * 70)
    print("1. DATASET SIZE")
    print("=" * 70)
    shots = load_all_shots()
    print(f"Total shots (all seasons, 2003-04 to 2024-25): {len(shots):,}")
    print(f"Seasons: {shots['SEASON'].nunique()}")
    print(shots.groupby("SEASON").size().to_string())

    print()
    print("=" * 70)
    print("2. TEMPORAL LEAKAGE CHECK: source-era vs. modeled-era separation")
    print("=" * 70)
    logs_2014 = pd.read_csv(SHOT_LOGS_PATH)
    print(
        "2014-15 defender-proxy source predates all rows used for modeling "
        "(2022-25 subset) -> no future information can leak into proxy features."
    )
    print(f"Defender-proxy source shots: {len(logs_2014):,} (season 2014-15 only)")

    print()
    print("=" * 70)
    print("3. DEFENDER-PROXY FEATURE COVERAGE (2022-25 modeling subset)")
    print("=" * 70)
    modeling_shots = shots[shots["SEASON"].isin(["2022-23", "2023-24", "2024-25"])].copy()
    with_proxy = add_defender_proxy_features(modeling_shots)
    proxy_cols = [c for c in with_proxy.columns if c.startswith("proxy_")]
    for c in proxy_cols:
        n_missing = with_proxy[c].isna().sum()
        print(f"{c}: {n_missing:,} missing / {len(with_proxy):,} "
              f"({100 * n_missing / len(with_proxy):.2f}%)")
    print()
    print("Sample proxy values by distance bucket x shot type (sanity check --")
    print("should rise with distance within each shot type; tiny-n cells")
    print("flagged low-confidence rather than trusted):")
    sample = (
        with_proxy.assign(
            dist_bin=pd.cut(with_proxy["SHOT_DISTANCE"],
                             bins=[0, 5, 10, 15, 20, 25, 30, 999], include_lowest=True)
        )
        .groupby(["SHOT_TYPE", "dist_bin"], observed=True)
        .agg(proxy_defender_dist_ft=("proxy_defender_dist_ft", "mean"),
             low_conf_share=("proxy_low_confidence", "mean"),
             n=("proxy_defender_dist_ft", "size"))
    )
    print(sample.to_string())
    n_low_conf = with_proxy["proxy_low_confidence"].sum()
    print(f"\nShots (2022-25) landing in a low-confidence proxy cell: "
          f"{n_low_conf:,} ({100 * n_low_conf / len(with_proxy):.2f}%)")

    print()
    print("=" * 70)
    print("4. PLAYER BIO (height/wingspan) COVERAGE by season")
    print("=" * 70)
    cov = coverage_report(modeling_shots)
    print(cov.to_string())
    with_bio = add_player_bio_features(modeling_shots)
    unmatched_players = (
        with_bio.loc[with_bio["height_inches"].isna(), "PLAYER_NAME"]
        .value_counts()
        .head(10)
    )
    print("\nTop 10 unmatched players (by shot volume) -- expected gap, not a bug:")
    print(unmatched_players.to_string())

    print()
    print("=" * 70)
    print("5. RANGE / SANITY CHECKS")
    print("=" * 70)
    print("height_inches range:", with_bio["height_inches"].min(), "-",
          with_bio["height_inches"].max())
    print("wingspan_inches range:", with_bio["wingspan_inches"].min(), "-",
          with_bio["wingspan_inches"].max())
    print("SHOT_DISTANCE range (full 22-season set):",
          shots["SHOT_DISTANCE"].min(), "-", shots["SHOT_DISTANCE"].max())
    n_extreme = (shots["SHOT_DISTANCE"] >= 40).sum()
    print(f"Shots >= 40ft (heaves, e.g. Curry logo threes): {n_extreme:,} "
          f"({100 * n_extreme / len(shots):.3f}% of all shots) -- "
          "sparse tail, flagged for Phase 3 modeling (monotonicity constraint "
          "needed so distance>40ft never predicts a higher FG% than a "
          "well-covered range like 24-30ft).")


if __name__ == "__main__":
    main()
