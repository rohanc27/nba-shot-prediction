# Distance Bucketing + Distance-Stratified Priors: Results

## Metrics: before (proxy+bio only) vs. after (+ distance buckets + stratified priors)

| Metric | Before this task | After this task | Change |
|---|---|---|---|
| LogReg AUC | 0.6552 | 0.6552 | 0.0000 |
| LogReg Log Loss | 0.6431 | 0.6429 | -0.0002 |
| LogReg Brier | 0.2265 | 0.2265 | 0.0000 |
| XGBoost baseline AUC | 0.6530 | 0.6553 | **+0.0023** |
| XGBoost Tuned AUC | 0.6586 | 0.6583 | -0.0003 |
| XGBoost Tuned Log Loss | 0.6401 | 0.6401 | 0.0000 |
| Test set size | 219,010 | 219,010 | unchanged |

**AUC target was 0.66 → 0.70+. Actual: best result is XGBoost baseline at
0.6553, a +0.0023 movement.** This is the largest single movement seen
across either round of feature work on this project so far, but still an
order of magnitude short of the target, and the tuned model (the one
actually deployed to `models/xgb_final.joblib`) didn't move at all.

## Edge case predictions

See `TEST_EDGE_CASES.md` for full methodology and table. Summary against
the stated success criteria:

| Criterion | Target | Result | Met? |
|---|---|---|---|
| Curry 50-footer | <10% | 26.7% (best case) | **No** |
| Curry normal 3PT (26ft) | 40%+ | 41.5-46.7% (3/4 models) | Mostly |
| Average player layup (clean control) | 60%+ | 66.9-68.9% | **Yes** |
| Average player deep 3 (32ft) | 25-30% | 16.2-24.5% | **No** |

The layup and normal-3PT cases look sane. The two failures matter more:
Curry's heave stays far above the target, and the deep-3 case undershoots
in the opposite direction across all four models.

## Feature importance: did the new features rank in the top 20?

**No.** Per `reports/shap_feature_importance.csv` (187 total encoded features):

- The 6 `distance_*_ft` binary flags: near-zero for XGBoost (5 of 6 exactly
  0.0, one at 0.000356; ranks 89-185 of 187). Expected and flagged in
  advance (commit 1) -- XGBoost can already split continuous `SHOT_DISTANCE`
  at any threshold, so explicit bucket dummies add nothing a tree model
  couldn't already do. These were always a LogReg-specific fix by design,
  not something meant to move XGBoost's SHAP ranking.
- The 12 distance-stratified prior columns: real but modest importance,
  ranks 33-67 (`player_prior_fg_pct_16_24` highest at #33). None crack the
  top 20; `SHOT_DISTANCE` itself, `player_prior_zone_fg_pct`, and `is_dunk`
  still dominate.

**Neither half of the success criterion ("distance buckets + distance-
stratified priors rank in top 20 by SHAP importance") was met.**

## Did normal shot accuracy improve?

Marginally, and only for XGBoost. The +0.0023 AUC gain on the baseline
XGBoost model is real (same test set, same split, only the feature set
changed) but small, and didn't survive into the tuned/frozen model. LogReg
saw no aggregate improvement at all -- consistent with the distance-bucket
dummies having no effect on a model whose main lever (the `SHOT_DISTANCE`
coefficient) already captured most of that signal, and the stratified
priors only meaningfully differing from the aggregate prior for players
with real bucket-specific history, which is most of the distribution but
not enough to move an AUC computed over 219,010 test shots dominated by
close/mid-range attempts.

## Bottom line

AUC did not reach the target range, and the core motivating edge case
(Curry's heave) is not fixed to the stated threshold. Full diagnosis of
*why*, and concrete next steps, are in the final-analysis section below.
