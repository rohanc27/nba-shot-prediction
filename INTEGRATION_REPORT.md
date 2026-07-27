# Integration Report: Defender-Proxy + Player-Bio Features

## Result

| Metric | Old (2022-25 only, no proxy) | New (proxy + bio features, same train/test split) |
|---|---|---|
| LogReg AUC | 0.6550 | 0.6552 |
| LogReg Log Loss | 0.6436 | 0.6431 |
| LogReg Brier | 0.2268 | 0.2265 |
| XGBoost baseline AUC | 0.6529 | 0.6530 |
| XGBoost Tuned AUC | 0.6584 | 0.6586 |
| XGBoost Tuned Log Loss | 0.6402 | 0.6401 |
| Test set size | 219,010 (2024-25) | 219,010 (2024-25) |
| Raw shots available | 654,948 (3 seasons) | 4,441,452 (22 seasons, after backcourt filter) |

**AUC did not meaningfully improve. Every model moved by +0.0001 to +0.0002 — noise, not signal.** The goal stated at the start of this work was AUC 0.66 → 0.80+; this integration gets nowhere close, and the two sections below explain why, mechanically, rather than just reporting the number.

## Why the new features didn't move AUC

**1. The defender-distance proxy is a deterministic function of features the model already has.**
`proxy_defender_dist_ft`/`proxy_shot_clock_sec`/`proxy_dribbles`/`proxy_touch_time_sec` are built by bucketing `SHOT_DISTANCE` and `SHOT_TYPE` and looking up a historical average — but `SHOT_DISTANCE` and `is_three` (derived from `SHOT_TYPE`) are already direct inputs to both models. A feature that is a coarse, lossy transform of two columns the model can already see head-on cannot add information; at best it's redundant, at worst it's a noisier version of what's already there. SHAP confirms this: `proxy_defender_dist_ft` ranks 43rd of 169 encoded features by importance, `proxy_touch_time_sec` ranks 68th. This was foreseeable from the feature's construction, and in hindsight should have been flagged more strongly during the data-acquisition phase — a per-shot-distance-bucket average was never going to substitute for real per-shot defender tracking, and I should have been explicit up front that this proxy's realistic ceiling was "near zero net-new information," not "a step toward defender-aware predictions."

**2. Height/wingspan carry real but small independent signal.**
Unlike the proxy features, `height_inches`/`wingspan_inches` are genuinely new information not derivable from existing columns. SHAP ranks them higher than the proxy features (24th and 26th of 169) — so they are contributing *something* — but the effect is small. Likely explanation: a tall player's finishing ability is already substantially captured by their `player_prior_*` features (a good rim finisher already shows up as high `player_prior_zone_fg_pct` for `Restricted Area`), so raw height/wingspan mostly duplicates information the model already has indirectly, leaving only a small residual signal (e.g. contest-avoidance on jumpers) for the new columns to explain.

**3. The training set itself did not actually expand.**
This is the most important mechanical reason the headline "4.45M shots" didn't move the needle: `train_logreg.py`/`train_xgb.py`/`tune_xgb.py` still hardcode `train_seasons = ["2022-23", "2023-24"]`, `test_seasons = ["2024-25"]` — unchanged from before, since the task scope was "add new features to the feature lists," not "change the train/test split." The model is trained on the same 435,082 rows and tested on the same 219,010 rows as before. The other ~4M shots (2003-04 through 2021-22) only entered the pipeline as extra history feeding the expanding-window player/team/zone priors computed by `build_features.py` — they were never shown to the classifiers as training examples. If the real goal is to test whether more training data raises the AUC ceiling, that requires actually widening `train_seasons`, which this integration did not do (see Follow-ups).

## A pre-existing issue this expansion made more consequential

`src/features/player_features.py` (and the team/opponent/action-prior modules) compute their Bayesian-smoothing prior means — `league_fg_pct`, `zone_means`, `action_means`, etc. — as a `.mean()` over the *entire* dataframe passed to `build_features()`, not a prior-only expanding window. This was already a data-leakage bug before this session (documented in an earlier portfolio review of this repo) but its scope just grew from "leaks across 3 seasons" to "leaks across 22 seasons" — every shot's smoothing prior is now influenced by outcomes from up to 21 years in the future or past relative to that shot. This wasn't in scope for this task (which asked for feature integration + retraining, not a leakage fix), so it was left as-is, but it means the reported AUCs — old and new alike — are not a clean, leakage-free number. Fixing it is flagged as a follow-up below; it was out of scope here and not touched.

## Issues encountered (and fixed) during integration

- `StandardScaler`/`LogisticRegression` can't handle the ~8-17% NaN rate in `height_inches`/`wingspan_inches` (players missing from the 2024-25 bio snapshot) — added a `SimpleImputer(strategy="median")` ahead of scaling in `train_logreg.py`. XGBoost needed no change; it handles NaN natively.
- `tune_xgb.py`, `evaluate_models.py`, and `explain_xgb.py` each duplicate their own feature-list constant rather than importing from `train_xgb.py` — all three needed the same 7-column addition independently, or they'd have crashed or silently ignored the new features.
- `models/*.joblib` and their `metrics.json` are gitignored except `models/xgb_final.joblib` (tracked deliberately, presumably because the Streamlit app loads it directly). Before/after metrics for the untracked artifacts are preserved as `reports/pre_expansion_baseline/` and `reports/post_expansion_metrics/` so the comparison survives in git history.
- `shap` wasn't installed in this environment; installed it to run `explain_xgb.py`. Worth adding to `requirements.txt` if SHAP explanation is meant to be part of the standard pipeline.
- No NaN, duplicate-row, or schema issues found in the expanded 4.45M-shot dataset (see `reports/data_expansion_validation.txt`); player-bio match rate is 88.9% for the 2022-25 modeling subset (vs. 33.8% across all 22 seasons combined, since the bio snapshot is 2024-25-only — expected, not a bug).

## What would actually move AUC toward 0.80

In descending order of expected impact:
1. **Real per-shot defender distance for 2022-25.** Doesn't exist publicly (confirmed during the data-acquisition phase); the 2014-15 proxy structurally cannot substitute for it, as shown above. This is the single biggest ceiling on this dataset, and no amount of feature engineering on the public schema removes it.
2. **Actually widen `train_seasons`** to use more of the 22 available seasons for training, not just as prior-history context. This is a real, cheap experiment that hasn't been run yet — it's a one-line change per training script, but changes what's being measured (more training data vs. more prior-history depth), so it wasn't bundled into this task without being asked.
3. **Fix the global-mean leakage** in `player_features.py`/`team_features.py`/`opponent_features.py` so smoothing priors only use past data — this would very likely *lower* the reported AUC slightly (removing an inflation source), giving a more honest number to build further improvements on top of.
4. Monotonicity handling for the sparse >40ft tail (12,534 shots, 0.28% of data) — the original ask about Steph Curry 50-footers not showing 40% FG% is a modeling/regularization question (e.g. isotonic constraint or explicit distance-decay term), not something more features fix.

None of the above were performed here — they're follow-up recommendations, not part of this integration.
