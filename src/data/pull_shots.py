"""Download NBA shot data from DomSamangy/NBA_Shots_04_25."""
from __future__ import annotations

import argparse
import io
import zipfile
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

DEFAULT_SEASONS = ["2022-23", "2023-24", "2024-25"]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

BASE_URL = (
    "https://github.com/DomSamangy/NBA_Shots_04_25/raw/main/"
    "NBA_{year}_Shots.csv.zip"
)


def season_to_year(season: str) -> int:
    """Convert '2024-25' -> 2025 (the season-ending year)."""
    start, end = season.split("-")
    start_year = int(start)
    end_year = (start_year // 100) * 100 + int(end)
    if end_year < start_year:
        end_year += 100
    return end_year


def download_season(season: str) -> pd.DataFrame:
    """Download and unzip one season's CSV."""
    year = season_to_year(season)
    url = BASE_URL.format(year=year)

    print(f"  Downloading {url}")
    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    buffer = io.BytesIO()
    with tqdm(total=total_size, unit="B", unit_scale=True, desc=f"  {season}") as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            buffer.write(chunk)
            pbar.update(len(chunk))

    buffer.seek(0)
    with zipfile.ZipFile(buffer) as zf:
        csv_name = zf.namelist()[0]
        with zf.open(csv_name) as f:
            df = pd.read_csv(f)

    return df


def normalize_columns(df: pd.DataFrame, season: str) -> pd.DataFrame:
    """Normalize types and add season column."""
    df = df.copy()
    df["SEASON"] = season

    if df["SHOT_MADE"].dtype == bool:
        df["SHOT_MADE"] = df["SHOT_MADE"].astype(int)
    elif df["SHOT_MADE"].dtype == object:
        df["SHOT_MADE"] = (
            df["SHOT_MADE"].astype(str).str.upper().map({"TRUE": 1, "FALSE": 0})
        )

    if "GAME_DATE" in df.columns:
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")

    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", nargs="+", default=DEFAULT_SEASONS)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    for season in args.seasons:
        out_path = RAW_DIR / f"shots_{season.replace('-', '_')}.parquet"
        if out_path.exists() and not args.force:
            existing = pd.read_parquet(out_path)
            print(f"OK {season}: {len(existing):,} shots already in {out_path.name}")
            continue

        print(f"\nDownloading {season}...")
        df = download_season(season)
        df = normalize_columns(df, season)
        df.to_parquet(out_path, index=False)

        made = int(df["SHOT_MADE"].sum())
        total = len(df)
        print(
            f"OK {season}: pulled {total:,} shots "
            f"({made:,} made, {total - made:,} missed, "
            f"{100 * made / total:.1f}% FG%) -> {out_path.name}"
        )


if __name__ == "__main__":
    main()
