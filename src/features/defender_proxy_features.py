"""Proxy defender/shot-context features built from the 2014-15 tracking log.

No public source has per-shot defender distance, shot clock, dribbles, or
touch time for the 2022-25 seasons this project actually models. What we
do have is a real, measured 2014-15 tracking dataset (128K shots). This
module aggregates that dataset into a lookup table -- average defender
distance etc. conditioned on shot distance and shot type -- and joins it
onto current-era shots as a historical PRIOR, not a measured value.

This is an approximation and should always be labeled as such downstream
(e.g. column names are prefixed `proxy_`). It captures "how far a
defender typically stands for a shot like this one" rather than "how far
the defender stood on this specific possession." It cannot fix the
absence of real per-shot defender tracking, but it's a strictly better
prior than omitting defender context entirely, and it introduces no
leakage risk since the source data (2014-15) entirely predates every
season in the current model (2022-25).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SHOT_LOGS_PATH = PROJECT_ROOT / "data" / "raw" / "external" / "shot_logs_2014_15.csv"
LOOKUP_PATH = PROJECT_ROOT / "data" / "processed" / "defender_proxy_lookup.parquet"

# 2ft bins from 0-30ft, then one bucket for everything beyond (heaves, etc).
DIST_BINS = list(range(0, 32, 2))
DIST_LABELS = [f"{lo}-{lo + 2}" for lo in DIST_BINS[:-1]]

# Below this, a (distance-bucket, shot-type) cell is dominated by rare/likely
# mislabeled combos in the source (e.g. a "3PT" shot logged at 2ft) rather
# than a real population of shots -- the mean in that cell is noise, not signal.
MIN_RELIABLE_N = 100


def _clean_shot_logs(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # TOUCH_TIME has corrupt negative values in the source data (min -163.6s);
    # a shot can't have negative touch time, so treat those as missing.
    df.loc[df["TOUCH_TIME"] < 0, "TOUCH_TIME"] = np.nan
    # SHOT_CLOCK is null when the game clock (not shot clock) was the binding
    # constraint (last 24s of quarter) -- that's a real, meaningful category,
    # not missing data, so leave it null and handle it explicitly downstream.
    return df


def build_lookup(force: bool = False) -> pd.DataFrame:
    """Aggregate the 2014-15 log into a (dist_bucket, pts_type) lookup table."""
    if LOOKUP_PATH.exists() and not force:
        return pd.read_parquet(LOOKUP_PATH)

    df = pd.read_csv(SHOT_LOGS_PATH)
    df = _clean_shot_logs(df)

    df["dist_bucket"] = pd.cut(
        df["SHOT_DIST"], bins=DIST_BINS + [999], labels=DIST_LABELS + ["30+"],
        right=False,
    )

    lookup = df.groupby(["dist_bucket", "PTS_TYPE"], observed=True).agg(
        proxy_defender_dist_ft=("CLOSE_DEF_DIST", "mean"),
        proxy_shot_clock_sec=("SHOT_CLOCK", "mean"),
        proxy_dribbles=("DRIBBLES", "mean"),
        proxy_touch_time_sec=("TOUCH_TIME", "mean"),
        proxy_n_source_shots=("SHOT_DIST", "size"),
    ).reset_index()

    lookup["proxy_low_confidence"] = lookup["proxy_n_source_shots"] < MIN_RELIABLE_N

    LOOKUP_PATH.parent.mkdir(parents=True, exist_ok=True)
    lookup.to_parquet(LOOKUP_PATH, index=False)
    return lookup


def add_defender_proxy_features(shots: pd.DataFrame) -> pd.DataFrame:
    """Join proxy defender/shot-context features onto a shots dataframe.

    Expects `shots` to have SHOT_DISTANCE and SHOT_TYPE columns (the
    DomSamangy schema). Buckets SHOT_DISTANCE the same way as the lookup
    table and maps SHOT_TYPE ("2PT Field Goal"/"3PT Field Goal") to the
    lookup's PTS_TYPE (2/3).
    """
    lookup = build_lookup()
    shots = shots.copy()

    shots["dist_bucket"] = pd.cut(
        shots["SHOT_DISTANCE"], bins=DIST_BINS + [999],
        labels=DIST_LABELS + ["30+"], right=False,
    )
    shots["PTS_TYPE"] = shots["SHOT_TYPE"].map(
        {"2PT Field Goal": 2, "3PT Field Goal": 3}
    )

    merged = shots.merge(lookup, on=["dist_bucket", "PTS_TYPE"], how="left")
    merged = merged.drop(columns=["dist_bucket", "PTS_TYPE"])
    return merged
