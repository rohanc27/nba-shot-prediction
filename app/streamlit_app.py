"""Streamlit app for NBA shot make probability prediction.

Run:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "xgb_final.joblib"
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "shots_features.parquet"

FEATURES = [
    "SHOT_DISTANCE",
    "LOC_X",
    "LOC_Y",
    "shot_angle_deg",
    "abs_angle_deg",
    "seconds_remaining_in_quarter",
    "QUARTER",
    "player_prior_fg_pct",
    "player_prior_zone_fg_pct",
    "player_prior_zone_name_fg_pct",
    "player_prior_zone_range_fg_pct",
    "player_prior_action_fg_pct",
    "player_prior_2pt_pct",
    "player_prior_3pt_pct",
    "player_prior_shots",
    "player_prior_zone_shots",
    "player_prior_zone_name_shots",
    "player_prior_zone_range_shots",
    "player_prior_action_shots",
    "player_prior_2pt_shots",
    "player_prior_3pt_shots",
    "team_prior_fg_pct",
    "team_prior_zone_fg_pct",
    "team_prior_2pt_pct",
    "team_prior_3pt_pct",
    "team_prior_shots",
    "team_prior_zone_shots",
    "team_prior_2pt_shots",
    "team_prior_3pt_shots",
    "is_home",
    "opponent_allowed_fg_pct",
    "opponent_allowed_zone_fg_pct",
    "opponent_allowed_2pt_pct",
    "opponent_allowed_3pt_pct",
    "opponent_allowed_shots",
    "opponent_allowed_zone_shots",
    "opponent_allowed_2pt_shots",
    "opponent_allowed_3pt_shots",
    "player_tendency_zone_rate",
    "player_tendency_action_rate",
    "player_tendency_shot_profile_rate",
    "player_zone_residual",
    "player_action_residual",
    "player_profile_residual",
    "is_three",
    "is_corner_3",
    "is_layup",
    "is_dunk",
    "is_late_clock",
    "is_overtime",
    "action_category",
    "BASIC_ZONE",
    "shot_profile",
    "zone_range",
]


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    df = pd.read_parquet(DATA_PATH)
    return df


def compute_shot_angle(loc_x: float, loc_y: float) -> float:
    return float(np.degrees(np.arctan2(loc_x, loc_y)))


def infer_basic_zone(shot_distance: float, loc_x: float, loc_y: float, is_three: int) -> str:
    if is_three:
        if abs(loc_x) > 220 and loc_y < 140:
            return "Right Corner 3" if loc_x > 0 else "Left Corner 3"
        return "Above the Break 3"

    if shot_distance < 8:
        return "Restricted Area"
    if shot_distance < 16:
        return "In The Paint (Non-RA)"
    return "Mid-Range"


def infer_zone_name(loc_x: float) -> str:
    if loc_x < -150:
        return "Left Side"
    if loc_x < -50:
        return "Left Side Center"
    if loc_x <= 50:
        return "Center"
    if loc_x <= 150:
        return "Right Side Center"
    return "Right Side"


def infer_zone_range(shot_distance: float) -> str:
    if shot_distance < 8:
        return "Less Than 8 ft."
    if shot_distance < 16:
        return "8-16 ft."
    if shot_distance < 24:
        return "16-24 ft."
    return "24+ ft."


def get_latest_player_row(df: pd.DataFrame, player_name: str) -> pd.Series:
    player_df = df[df["PLAYER_NAME"] == player_name].copy()
    if player_df.empty:
        raise ValueError(f"No rows found for player {player_name}")
    player_df = player_df.sort_values(["GAME_DATE", "GAME_ID", "QUARTER", "MINS_LEFT", "SECS_LEFT"])
    return player_df.iloc[-1]


def build_input_row(
    base_row: pd.Series,
    player_name: str,
    loc_x: float,
    loc_y: float,
    quarter: int,
    mins_left: int,
    secs_left: int,
    action_category: str,
    is_home: int,
) -> pd.DataFrame:
    row = base_row.copy()

    shot_distance = float(np.sqrt(loc_x**2 + loc_y**2) / 10)
    shot_angle = compute_shot_angle(loc_x, loc_y)

    is_three = int(shot_distance >= 22)
    basic_zone = infer_basic_zone(shot_distance, loc_x, loc_y, is_three)
    zone_name = infer_zone_name(loc_x)
    zone_range = infer_zone_range(shot_distance)

    row["PLAYER_NAME"] = player_name
    row["LOC_X"] = loc_x
    row["LOC_Y"] = loc_y
    row["SHOT_DISTANCE"] = shot_distance
    row["shot_angle_deg"] = shot_angle
    row["abs_angle_deg"] = abs(shot_angle)
    row["QUARTER"] = quarter
    row["MINS_LEFT"] = mins_left
    row["SECS_LEFT"] = secs_left
    row["seconds_remaining_in_quarter"] = mins_left * 60 + secs_left
    row["is_late_clock"] = int(row["seconds_remaining_in_quarter"] <= 24)
    row["is_overtime"] = int(quarter >= 5)
    row["is_three"] = is_three
    row["is_corner_3"] = int(basic_zone in ["Left Corner 3", "Right Corner 3"])
    row["BASIC_ZONE"] = basic_zone
    row["ZONE_NAME"] = zone_name
    row["ZONE_RANGE"] = zone_range
    row["action_category"] = action_category
    row["is_layup"] = int(action_category == "layup")
    row["is_dunk"] = int(action_category == "dunk")
    row["is_home"] = is_home

    row["shot_profile"] = f"{basic_zone} | {zone_range} | {action_category}"
    row["zone_range"] = f"{zone_name} | {zone_range}"

    return pd.DataFrame([row[FEATURES]])


def main():
    st.set_page_config(
        page_title="NBA Shot Probability Predictor",
        page_icon="🏀",
        layout="wide",
    )

    st.title("🏀 NBA Shot Probability Predictor")
    st.caption("Interactive expected field goal model using shot geometry, player history, team context, and game state.")

    model = load_model()
    df = load_data()

    players = sorted(df["PLAYER_NAME"].dropna().unique())
    actions = sorted(df["action_category"].dropna().unique())

    with st.sidebar:
        st.header("Shot setup")

        player = st.selectbox("Player", players, index=players.index("Stephen Curry") if "Stephen Curry" in players else 0)
        action = st.selectbox("Shot action", actions, index=actions.index("jump_shot") if "jump_shot" in actions else 0)

        quarter = st.slider("Quarter", 1, 5, 1)
        mins_left = st.slider("Minutes left", 0, 12, 6)
        secs_left = st.slider("Seconds left", 0, 59, 0)
        is_home = st.radio("Home or away", ["Home", "Away"], horizontal=True)
        is_home_value = 1 if is_home == "Home" else 0

        st.divider()
        st.subheader("Shot location")
        loc_x = st.slider("LOC_X", -250, 250, 0)
        loc_y = st.slider("LOC_Y", 0, 470, 80)

    base_row = get_latest_player_row(df, player)
    input_df = build_input_row(
        base_row=base_row,
        player_name=player,
        loc_x=loc_x,
        loc_y=loc_y,
        quarter=quarter,
        mins_left=mins_left,
        secs_left=secs_left,
        action_category=action,
        is_home=is_home_value,
    )

    probability = float(model.predict_proba(input_df)[0, 1])

    col1, col2 = st.columns([1.1, 1])

    with col1:
        st.subheader("Prediction")
        st.metric("Predicted make probability", f"{probability * 100:.1f}%")

        st.write("### Input summary")
        summary = pd.DataFrame({
            "Field": [
                "Player",
                "Action",
                "Shot distance",
                "Location",
                "Basic zone",
                "Zone range",
                "Quarter",
                "Clock",
                "Home/Away",
            ],
            "Value": [
                str(player),
                str(action),
                f"{input_df['SHOT_DISTANCE'].iloc[0]:.1f} ft",
                f"({loc_x}, {loc_y})",
                str(input_df["BASIC_ZONE"].iloc[0]),
                str(input_df["zone_range"].iloc[0]),
                str(quarter),
                f"{mins_left}:{secs_left:02d}",
                str(is_home),
            ],
        })

        st.dataframe(summary, width="stretch", hide_index=True)

    with col2:
        st.subheader("Simple court view")

        court_df = pd.DataFrame({
            "LOC_X": [loc_x],
            "LOC_Y": [loc_y],
            "Prediction": [probability],
        })

        st.scatter_chart(
            court_df,
            x="LOC_X",
            y="LOC_Y",
            size="Prediction",
            height=500,
        )

        st.caption("Current version uses sliders for shot location. Later we can upgrade this to clickable court selection.")

    with st.expander("Model notes"):
        st.write(
            """
            This prediction uses a trained XGBoost model. Features include shot geometry,
            player historical shooting priors, team offensive context, opponent defensive context,
            shot profile interactions, player shot tendencies, and residual player skill features.
            """
        )


if __name__ == "__main__":
    main()
