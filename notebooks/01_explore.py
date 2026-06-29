"""Quick exploration of the raw shot data."""
import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw")
paths = sorted(RAW_DIR.glob("shots_*.parquet"))

# Check column diff across seasons
print("=== Column diff across seasons ===")
col_sets = {}
for p in paths:
    df = pd.read_parquet(p)
    col_sets[p.stem] = set(df.columns)

all_cols = set().union(*col_sets.values())
for season, cols in col_sets.items():
    missing = all_cols - cols
    if missing:
        print(f"  {season} is missing: {sorted(missing)}")
    else:
        print(f"  {season}: all columns present")
print()

# Load combined data
df = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
print(f"Combined: {len(df):,} shots, {df['SEASON'].nunique()} seasons")
print()

print("=== Missing values (key columns only) ===")
key_cols = ["SHOT_MADE", "LOC_X", "LOC_Y", "SHOT_DISTANCE",
            "ACTION_TYPE", "SHOT_TYPE", "BASIC_ZONE", "QUARTER",
            "MINS_LEFT", "SECS_LEFT"]
nulls = df[key_cols].isnull().sum()
print(nulls[nulls > 0] if (nulls > 0).any() else "  None")
print()

print("=== SHOT_TYPE distribution ===")
print(df["SHOT_TYPE"].value_counts())
print()

print("=== BASIC_ZONE distribution + FG% ===")
zone_stats = df.groupby("BASIC_ZONE").agg(
    count=("SHOT_MADE", "count"),
    fg_pct=("SHOT_MADE", "mean"),
).sort_values("count", ascending=False)
print(zone_stats.to_string())
print()

print("=== ACTION_TYPE top 15 ===")
print(df["ACTION_TYPE"].value_counts().head(15))
print()

print(f"ACTION_TYPE total unique values: {df['ACTION_TYPE'].nunique()}")
print()

print("=== SHOT_DISTANCE summary ===")
print(df["SHOT_DISTANCE"].describe())
print()

print("=== QUARTER distribution ===")
print(df["QUARTER"].value_counts().sort_index())
print()

print("=== Coordinate ranges ===")
print(f"LOC_X: {df['LOC_X'].min():.1f} to {df['LOC_X'].max():.1f}")
print(f"LOC_Y: {df['LOC_Y'].min():.1f} to {df['LOC_Y'].max():.1f}")
print()

print("=== MINS_LEFT / SECS_LEFT ranges ===")
print(f"MINS_LEFT: {df['MINS_LEFT'].min()} to {df['MINS_LEFT'].max()}")
print(f"SECS_LEFT: {df['SECS_LEFT'].min()} to {df['SECS_LEFT'].max()}")
