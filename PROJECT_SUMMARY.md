# Project Summary

## Project

NBA Expected Field Goal Model

## Goal

Predict the probability that an NBA shot attempt will be made using information available before the shot.

## Dataset

Approximately 654,000 NBA shot attempts from the 2022-23, 2023-24, and 2024-25 seasons.

## Target

`SHOT_MADE`

## Split

- Train: 2022-23, 2023-24
- Test: 2024-25

## Final Model

Tuned XGBoost classifier

## Final Test Metrics

| Metric | Value |
|---|---:|
| Accuracy | 0.6302 |
| AUC | 0.6584 |
| Log loss | 0.6402 |
| Brier score | 0.2254 |

## Main Feature Families

- Shot geometry
- Action type
- Player historical priors
- Team offensive priors
- Opponent defensive priors
- Shot profile interactions
- Player shot tendencies
- Player residual skill
- Game state

## Main Outputs

- Trained models
- Evaluation tables
- Calibration plots
- ROC and PR curves
- SHAP explainability plots
- Error analysis reports
- Streamlit web app
- README documentation

## Final App

```bash
streamlit run app/streamlit_app.py
