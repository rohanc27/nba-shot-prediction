"""Download current-season NBA player height/wingspan data.

Height and wingspan are stable, near-permanent physical attributes for an
adult player, so a single current-season roster snapshot (2024-25) can
reasonably be joined onto shots from all three seasons (2022-25) used in
this project by player identity. Coverage gap: players who appear in the
2022-23/2023-24 data but not the 2024-25 season (retired, released, etc.)
will have no match and should be left as missing, not imputed with a
league-average fallback that would mask the gap.

Source: SCORE Sports Data Repository (data.scorenetwork.org), a public,
unauthenticated CSV -- no scraping of Basketball-Reference required (BR
blocks automated fetches with a 403).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "data" / "raw" / "external"
OUT_PATH = OUT_DIR / "player_bio_2024_25.csv"

URL = "https://data.scorenetwork.org/data/nba_wingspan_performance_2025.csv"

EXPECTED_COLUMNS = {"name", "height_inches", "wingspan_inches"}


def download(force: bool = False) -> pd.DataFrame:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if OUT_PATH.exists() and not force:
        print(f"OK: already downloaded -> {OUT_PATH}")
        return pd.read_csv(OUT_PATH)

    print(f"Downloading {URL}")
    response = requests.get(URL, timeout=60)
    response.raise_for_status()
    OUT_PATH.write_bytes(response.content)

    df = pd.read_csv(OUT_PATH)
    missing = EXPECTED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Downloaded file missing expected columns: {missing}")

    print(f"OK: {len(df):,} players, season 2024-25 -> {OUT_PATH}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    download(force=args.force)


if __name__ == "__main__":
    main()
