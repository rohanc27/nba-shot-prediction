# Edge Case Tests: Distance-Aware Features

Run via `python scripts/test_edge_cases.py`. Compares an in-memory
reconstruction of the OLD (pre-distance-bucketing) LogReg/XGBoost pipelines
against the actual saved NEW models (`models/logreg.joblib`, `models/xgb.joblib`),
on the same real shot rows wherever a real one exists.

**Methodology note:** wherever possible these are real shots pulled from the
2024-25 test set, not hand-built feature vectors — a synthetic row risks
combining features (e.g. a 50ft `SHOT_DISTANCE` next to a `LOC_Y` that
geometrically implies 25ft) that no real shot could ever have, which would
make the model's prediction meaningless. The one exception is the 50ft
heave: Curry's real max distance in the 2024-25 data is 40ft, so that case
overrides the distance fields on his real 40ft row (`SHOT_DISTANCE`,
`distance_35_plus_ft`, `distance_28_35_ft`) and leaves everything else
(`LOC_X`/`LOC_Y`/angle/priors) as they actually were on that shot — an
approximation, not a clean 50ft shot, and flagged as such in the results.

## Results

| Case | Actual | OLD LogReg | NEW LogReg | OLD XGB | NEW XGB |
|---|---|---|---|---|---|
| Curry, real 40ft heave | missed | 29.3% | 36.3% | 26.6% | 26.7% |
| Curry, synthetic 50ft heave | (n/a, approximation) | 30.5% | 32.3% | 26.6% | 26.7% |
| Plumlee, real 5ft layup (buzzer-beater, non-RA) | missed | 33.7% | 34.0% | 18.3% | 17.2% |
| Plumlee, real 0-2ft restricted-area finish (clean control) | made | 68.2% | 68.9% | 66.9% | 68.2% |
| Curry, real 26ft pull-up 3 | made | 46.0% | 46.7% | 37.5% | 41.5% |
| Pritchard, real 32ft deep pull-up 3 | missed | 24.4% | 24.5% | 16.2% | 16.9% |

## Against the stated success criteria

| Criterion | Target | Result | Met? |
|---|---|---|---|
| Curry 50-footer | <10% | 26.7% (best case, NEW XGB) | **No** |
| Curry normal 3PT (26ft) stays high | 40%+ | 41.5-46.7% (3 of 4 models); OLD XGB 37.5% | Mostly |
| Average player layup is high | 60%+ | 66.9-68.9% (clean control) | **Yes** |
| Average player deep 3 (32ft) | 25-30% | 16.2-24.5% | **No** (all 4 models undershoot) |

## Why the Plumlee "5ft layup" case needed a second, cleaner example

The first real 5ft shot found for a rotation big (Mason Plumlee) turned out to
be a late-clock (0 seconds left in the quarter), non-restricted-area
"Layup Shot" — a genuinely harder, contested shot, not a clean finish. His
own zone-specific prior there (35.5%) and league-wide FG% for that exact
`ACTION_TYPE` label (46.5%) are both well below his overall career average
(59.8%). The ~17-33% predictions on that row are the model correctly reading
rich context (zone, action type, game clock), not a bug — but it's also not
representative of "an average player's routine layup," so a second, cleaner
control (restricted area, non-late-clock, 0-2ft) was added and clears the
60%+ bar comfortably (66.9-68.9%).

## The headline failure: Curry's heave

Neither model gets close to the <10% target. Most notably, **XGBoost's
prediction is essentially unchanged old vs. new (26.6% → 26.7%)** —
despite the new `player_prior_fg_pct_35_plus` feature correctly showing
Curry's real 35+ft history (~12%, from 9 prior attempts, per the smoke test
in the `distance_stratified_player_priors.py` commit). The feature contains
the right information; the model isn't weighting it heavily enough for this
row. Full diagnosis and recommended next steps in `DISTANCE_BUCKETING_REPORT.md`.
