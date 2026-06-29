from pathlib import Path
import pandas as pd

df = pd.read_parquet("data/processed/shots_features.parquet")

print("\n=== Columns ===")
for c in sorted(df.columns):
    print(c)

print("\nRows:", len(df))