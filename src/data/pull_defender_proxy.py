"""Download the 2014-15 NBA shot-tracking log (SportVU era).

IMPORTANT CAVEAT: this is the only publicly available dataset with real
per-shot defender distance, shot clock, dribbles, and touch time. The NBA
stopped exposing this level of tracking data publicly after the 2014-15
season (and further restricted defender-distance splits on stats.nba.com
in Dec 2020), so no equivalent data exists for the 2022-25 seasons used
elsewhere in this project.

This dataset CANNOT be merged shot-for-shot with the 2022-25 data (wrong
era, different players/rules/pace). It is downloaded so it can be used to
build proxy/prior features (see src/features/defender_proxy_features.py)
-- e.g. "typical defender distance for a 25ft catch-and-shoot 3" -- which
are attached to current-era shots as a coarse historical prior, NOT as a
per-shot measurement. Treat any feature derived from this source as an
approximation, and label it as such everywhere it's used.

Source: originally published on Kaggle (dansbecker/nba-shot-logs), mirrored
as a raw CSV in this public GitHub repo:
https://github.com/tanmaychk/EDA-NBA-shot_logs
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "data" / "raw" / "external"
OUT_PATH = OUT_DIR / "shot_logs_2014_15.csv"

URL = (
    "https://raw.githubusercontent.com/tanmaychk/EDA-NBA-shot_logs/"
    "main/shot_logs.csv"
)

EXPECTED_COLUMNS = {
    "SHOT_CLOCK",
    "DRIBBLES",
    "TOUCH_TIME",
    "SHOT_DIST",
    "PTS_TYPE",
    "SHOT_RESULT",
    "CLOSE_DEF_DIST",
}


def download(force: bool = False) -> pd.DataFrame:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if OUT_PATH.exists() and not force:
        print(f"OK: already downloaded -> {OUT_PATH}")
        return pd.read_csv(OUT_PATH)

    print(f"Downloading {URL}")
    response = requests.get(URL, timeout=120)
    response.raise_for_status()
    OUT_PATH.write_bytes(response.content)

    df = pd.read_csv(OUT_PATH)
    missing = EXPECTED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Downloaded file missing expected columns: {missing}")

    print(
        f"OK: {len(df):,} shots, {df['player_name'].nunique()} players, "
        f"season 2014-15 -> {OUT_PATH}"
    )
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    download(force=args.force)


if __name__ == "__main__":
    main()
