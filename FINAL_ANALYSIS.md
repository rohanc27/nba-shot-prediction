# Final Analysis: Distance-Aware Features Impact

## Did AUC improve?

Only for XGBoost's baseline (untuned) model, and only modestly: 0.6530 →
0.6553 (+0.0023). LogReg didn't move (0.6552 → 0.6552). The tuned/frozen
model -- the one actually shipped -- went slightly backward (0.6586 →
0.6583, within noise). None of this approaches the 0.66 → 0.70+ target.

## Failure diagnosis (per the checklist in the task)

**1. SHAP ranks — are distance buckets used?**
Barely, and asymmetrically by design. The 6 `distance_*_ft` binary flags
are near-zero for XGBoost (ranks 89-185 of 187) -- exactly as predicted in
the commit 1 message, since a tree model can already split continuous
`SHOT_DISTANCE` at any threshold without needing explicit bucket dummies.
The 12 distance-stratified prior columns do carry real importance (ranks
33-67), more than the flags but still outside the top 20.

**2. Correlation between distance_*_ft columns (multicollinearity)?**
Yes, structurally: the 6 dummies sum to exactly 1 for every row (verified:
`test[cols].sum(axis=1).unique() == [1]`), which is the classic "dummy
variable trap" -- they're perfectly collinear with the model intercept.
L2-regularized LogisticRegression handles this numerically without
crashing, but it does mean coefficient weight can split arbitrarily
between the intercept and the 6 flags, diluting how much individual
importance any one bucket flag shows. This wasn't caught before
implementation and should have used 5 dummies + an implicit reference
category (e.g. drop `distance_16_24_ft` as baseline). It likely explains
some of why the flags look so unimportant in LogReg's coefficients, though
it doesn't explain XGBoost's near-zero SHAP (trees aren't affected by
linear collinearity the same way).

**3. Are distance-stratified priors mostly NaN for low-volume players?**
No NaN at all (0.00% across all 6 buckets, by design -- the fallback to
`player_prior_fg_pct` guarantees a real number). But *thin real history*
is a genuine issue for the two rarest buckets: conditioned on shots
actually in each bucket, median prior-shot-count is 452-2,204 for the four
buckets under 28ft, but only **88** for 28-35ft and just **4** for 35+ft.
16.6% of 35+ft shots are a player's first-ever attempt there (zero prior
history, pure fallback to their overall average that round). The 35+
bucket is also only 223 of 219,010 test shots (0.1%) -- individually
correct for the handful of cases it covers, but far too small a slice to
move an aggregate SHAP ranking or AUC.

**4. Have we hit the ceiling on shot-chart data without defender
distance/shot clock?**
Partially, but that's not the main story here -- this round wasn't about
defender data (that's the prior integration's finding). The more specific,
new finding from this round: **even with the theoretically correct
feature present, the model doesn't weight it heavily enough for the exact
row it was built to fix.** Curry's real `player_prior_fg_pct_35_plus` is
~12% (9 prior attempts, verified in the `distance_stratified_player_priors.py`
commit), yet XGBoost predicts 26.7% on his actual 40ft heave -- essentially
identical to the 26.6% the pre-distance-bucketing model predicted. The
right number exists in the feature set; the model isn't using it decisively
for this row. Two structural reasons why:

- **LogReg is additive.** `player_prior_fg_pct`, `player_prior_3pt_pct`
  (Curry's *overall* aggregates, ~45-47%) are still separate features,
  unchanged, still carrying large positive coefficients because they're
  useful for the other 99.9% of shots. Nothing in a plain linear model
  forces the bucket-specific prior to *override* the aggregate one for
  this row -- it just adds another term, which the aggregate terms'
  established weight can outvote for a single observation.
- **XGBoost sees too few 35+ft training examples to build decisive
  splits.** With ~a few hundred such shots total across 435,082 training
  rows, gradient boosting doesn't get a strong or consistent enough
  gradient signal from that feature to justify trees that heavily
  downweight a shot just because `player_prior_fg_pct_35_plus` is low --
  especially when the far more common features (zone, action type, general
  skill priors) already explain most of the loss on the bulk of the data.

The average-player deep-3 case (16.2-24.5% vs. a 25-30% target) is a
smaller version of the same story in the opposite direction: without a
context-specific reason to push the prediction down (late clock, poor
matchup, etc. -- whichever the real row happened to have), the model may
be *over*-crediting the bucket-level league prior at this range rather
than under-crediting it. Not diagnosed further here given scope, but
worth checking whether this specific row had unusual context (similar to
the Plumlee false-alarm case) before concluding it's a systematic bias.

## Next steps (recommended, not implemented -- out of this task's scope)

Ranked by expected impact on the Curry-heave failure specifically:

1. **Explicit interaction features**, e.g. `distance_35_plus_ft *
   player_prior_fg_pct_35_plus`, or better, *replace* the aggregate
   `player_prior_fg_pct`/`player_prior_3pt_pct` with the matching bucket's
   value entirely for that row (rather than adding the bucket prior
   alongside the aggregate one) -- this directly targets the additive-model
   problem in LogReg.
2. **XGBoost `monotone_constraints`** on `SHOT_DISTANCE`, forcing predicted
   make-probability to be non-increasing with distance. This is a built-in
   XGBoost feature, directly addresses the exact failure mode observed
   here, and doesn't require new data.
3. **Fix the dummy-variable trap** (drop one bucket as reference category)
   -- cheap, standard practice, probably a small but real improvement to
   LogReg's coefficient stability and interpretability.
4. Re-diagnose the deep-3 undershoot with the same row-level scrutiny used
   for the Plumlee case, before assuming it's systematic.

None of the above were implemented in this task -- they're flagged as
follow-ups per the failure-diagnosis instructions, not executed.
