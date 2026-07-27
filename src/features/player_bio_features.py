"""Join real (not proxied) player height/wingspan onto shots by player name.

Height and wingspan don't meaningfully change season to season for an
adult NBA player, so a single 2024-25 roster snapshot is used as a static
per-player attribute across all seasons in the shots data. Coverage is
necessarily incomplete: players active in 2022-23/2023-24 but not 2024-25
(retired, out of the league, etc.) will have no match. Those rows are left
as NaN -- do not silently backfill with a league-average, since that would
hide the coverage gap rather than surface it.
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BIO_PATH = PROJECT_ROOT / "data" / "raw" / "external" / "player_bio_2024_25.csv"

_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def _strip_diacritics(name: str) -> str:
    # e.g. "Dončić" -> "Doncic", "Şengün" -> "Sengun" -- both sources use
    # inconsistent diacritic conventions, so ASCII-fold before comparing.
    normalized = unicodedata.normalize("NFKD", name)
    return "".join(c for c in normalized if not unicodedata.combining(c))


def _normalize_name(name: str) -> str:
    name = _strip_diacritics(name).lower().strip()
    name = re.sub(r"[.\-']", "", name)
    tokens = [t for t in name.split() if t not in _SUFFIXES]
    return " ".join(tokens)


def load_player_bio() -> pd.DataFrame:
    bio = pd.read_csv(BIO_PATH)
    bio["_join_key"] = bio["name"].map(_normalize_name)
    dupes = bio["_join_key"][bio["_join_key"].duplicated()]
    if not dupes.empty:
        raise ValueError(f"Ambiguous normalized names in player bio: {dupes.tolist()}")
    return bio[["_join_key", "height_inches", "wingspan_inches"]]


def add_player_bio_features(shots: pd.DataFrame, name_col: str = "PLAYER_NAME") -> pd.DataFrame:
    bio = load_player_bio()
    shots = shots.copy()
    shots["_join_key"] = shots[name_col].map(_normalize_name)

    merged = shots.merge(bio, on="_join_key", how="left")
    merged = merged.drop(columns=["_join_key"])
    merged["wingspan_minus_height_in"] = (
        merged["wingspan_inches"] - merged["height_inches"]
    )
    return merged


def coverage_report(shots: pd.DataFrame, name_col: str = "PLAYER_NAME") -> pd.DataFrame:
    """Per-season match-rate report: what fraction of shots got a bio match."""
    merged = add_player_bio_features(shots, name_col=name_col)
    matched = merged["height_inches"].notna()
    out = (
        merged.assign(matched=matched)
        .groupby("SEASON")["matched"]
        .agg(["mean", "size"])
        .rename(columns={"mean": "match_rate", "size": "n_shots"})
    )
    return out
