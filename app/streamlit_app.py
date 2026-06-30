"""Streamlit app for NBA shot make probability prediction.

Run:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "xgb_final.joblib"
PLAYER_LOOKUP_PATH = PROJECT_ROOT / "data" / "processed" / "app_player_lookup.parquet"
LEAGUE_CONTEXT_PATH = PROJECT_ROOT / "data" / "processed" / "app_league_context.parquet"
LEAGUE_ZONE_PATH = PROJECT_ROOT / "data" / "processed" / "app_league_zone.parquet"

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
    player_lookup = pd.read_parquet(PLAYER_LOOKUP_PATH)
    league_context = pd.read_parquet(LEAGUE_CONTEXT_PATH)
    league_zone = pd.read_parquet(LEAGUE_ZONE_PATH)
    return player_lookup, league_context, league_zone


def add_css() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 1rem;
            max-width: 95%;
        }

        h1 {
            font-size: 3rem !important;
        }

        h2, h3 {
            margin-top: 0.4rem;
        }

        div[data-testid="stMetricValue"] {
            font-size: 3.2rem;
        }

        div[data-testid="stMetricValue"] > div {
            overflow: visible;
            white-space: nowrap;
        }

        .prediction-card {
            border: 1px solid rgba(250,250,250,0.15);
            border-radius: 18px;
            padding: 28px;
            background: rgba(255,255,255,0.035);
            text-align: center;
            margin-bottom: 16px;
        }

        .big-prob {
            font-size: 4.4rem;
            font-weight: 800;
            line-height: 1;
        }

        .subtle {
            color: rgba(250,250,250,0.65);
            font-size: 0.95rem;
        }

        .good {
            color: #21c55d;
            font-weight: 700;
        }

        .bad {
            color: #ef4444;
            font-weight: 700;
        }

        .info-card {
            border: 1px solid rgba(250,250,250,0.12);
            border-radius: 14px;
            padding: 18px;
            background: rgba(255,255,255,0.025);
            height: 100%;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


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
    player_df = player_df.sort_values(
        ["GAME_DATE", "GAME_ID", "QUARTER", "MINS_LEFT", "SECS_LEFT"]
    )
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


def draw_half_court(loc_x: float, loc_y: float, probability: float):
    fig, ax = plt.subplots(figsize=(7, 6))

    # Court boundary
    ax.add_patch(patches.Rectangle((-250, 0), 500, 470, fill=False, linewidth=2))

    # Hoop and backboard
    ax.add_patch(patches.Circle((0, 52.5), radius=7.5, fill=False, linewidth=2))
    ax.plot([-30, 30], [40, 40], linewidth=2)

    # Paint
    ax.add_patch(patches.Rectangle((-80, 0), 160, 190, fill=False, linewidth=2))
    ax.add_patch(patches.Rectangle((-60, 0), 120, 190, fill=False, linewidth=1))

    # Free throw circle
    ax.add_patch(patches.Circle((0, 190), radius=60, fill=False, linewidth=2))

    # Restricted area
    ax.add_patch(patches.Arc((0, 52.5), 80, 80, theta1=0, theta2=180, linewidth=2))

    # Three point line
    ax.plot([-220, -220], [0, 140], linewidth=2)
    ax.plot([220, 220], [0, 140], linewidth=2)
    ax.add_patch(patches.Arc((0, 52.5), 475, 475, theta1=22, theta2=158, linewidth=2))

    # Shot marker
    ax.scatter(
        [loc_x],
        [loc_y],
        s=450,
        alpha=0.9,
        edgecolors="black",
        linewidth=1.5,
    )
    ax.text(
        loc_x,
        loc_y + 18,
        f"{probability * 100:.1f}%",
        ha="center",
        fontsize=12,
        weight="bold",
    )

    ax.set_xlim(-260, 260)
    ax.set_ylim(0, 480)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Shot location", fontsize=16, weight="bold")

    return fig


def player_summary(df: pd.DataFrame, player: str) -> dict:
    row = df[df["PLAYER_NAME"] == player].iloc[0]
    return {
        "shots": int(row["app_shots"]),
        "fg_pct": float(row["app_fg_pct"]),
        "three_pct": float(row["app_three_pct"]),
        "rim_pct": float(row["app_rim_pct"]),
    }

def league_average_for_context(
    league_context: pd.DataFrame,
    league_zone: pd.DataFrame,
    input_df: pd.DataFrame,
) -> float:
    zone = input_df["BASIC_ZONE"].iloc[0]
    action = input_df["action_category"].iloc[0]

    context = league_context[
        (league_context["BASIC_ZONE"] == zone)
        & (league_context["action_category"] == action)
    ]

    if len(context) > 0 and int(context["league_context_shots"].iloc[0]) >= 100:
        return float(context["league_context_fg_pct"].iloc[0])

    zone_context = league_zone[league_zone["BASIC_ZONE"] == zone]
    if len(zone_context) > 0:
        return float(zone_context["league_zone_fg_pct"].iloc[0])

    return 0.47

def build_explanation(input_df: pd.DataFrame, probability: float, league_avg: float):
    row = input_df.iloc[0]
    explanations = []

    if row["SHOT_DISTANCE"] < 8:
        explanations.append(("Close to basket", "Boosts make probability"))
    elif row["SHOT_DISTANCE"] >= 24:
        explanations.append(("Long-distance shot", "Lowers make probability"))

    if row["is_dunk"] == 1:
        explanations.append(("Dunk attempt", "Very high-value shot profile"))

    if row["is_late_clock"] == 1:
        explanations.append(("Late-clock situation", "Usually lowers shot quality"))

    if row["player_prior_zone_fg_pct"] > league_avg:
        explanations.append(("Player zone history", "Shooter has strong prior results from this zone"))

    if row["player_action_residual"] > 0:
        explanations.append(("Player action skill", "Shooter is above league average for this action type"))

    if probability > league_avg:
        explanations.append(("Model vs. league average", "Prediction is above comparable league-average shot quality"))
    else:
        explanations.append(("Model vs. league average", "Prediction is below comparable league-average shot quality"))

    return explanations[:5]


def main():
    st.set_page_config(
        page_title="NBA Shot Probability Predictor",
        page_icon="🏀",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    add_css()

    st.title("🏀 NBA Shot Probability Predictor")
    st.caption(
        "Interactive expected field goal model using shot geometry, player history, team context, and game state."
    )

    model = load_model()
    df, league_context, league_zone = load_data()

    players = sorted(df["PLAYER_NAME"].dropna().unique())
    actions = sorted(df["action_category"].dropna().unique())

    with st.sidebar:
        st.header("Shot setup")

        default_player = players.index("Stephen Curry") if "Stephen Curry" in players else 0
        player = st.selectbox("Player", players, index=default_player)

        default_action = actions.index("jump_shot") if "jump_shot" in actions else 0
        action = st.selectbox("Shot action", actions, index=default_action)

        quarter = st.slider("Quarter", 1, 5, 1)
        mins_left = st.slider("Minutes left", 0, 12, 6)
        secs_left = st.slider("Seconds left", 0, 59, 0)
        home_choice = st.radio("Home or away", ["Home", "Away"], horizontal=True)
        is_home_value = 1 if home_choice == "Home" else 0

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
    league_avg = league_average_for_context(league_context, league_zone, input_df)
    diff = probability - league_avg
    psummary = player_summary(df, player)

    top_left, top_right = st.columns([1, 1.15])

    with top_left:
        st.markdown(
            f"""
            <div class="prediction-card">
                <div class="subtle">Predicted make probability</div>
                <div class="big-prob">{probability * 100:.1f}%</div>
                <div class="subtle">Comparable league average: {league_avg * 100:.1f}%</div>
                <div class="{'good' if diff >= 0 else 'bad'}">
                    {diff * 100:+.1f} percentage points vs. league average
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        c1, c2, c3 = st.columns(3)
        c1.metric("Shot distance", f"{input_df['SHOT_DISTANCE'].iloc[0]:.1f} ft")
        ZONE_ABBREV = {
            "In The Paint (Non-RA)": "Paint",
            "Restricted Area": "Rim",
            "Mid-Range": "Mid-Range",
            "Above the Break 3": "Above Break 3",
            "Left Corner 3": "Left Corner 3",
            "Right Corner 3": "Right Corner 3",
        }

        c2.metric("Zone", ZONE_ABBREV.get(input_df["BASIC_ZONE"].iloc[0], input_df["BASIC_ZONE"].iloc[0]))
        c3.metric("Clock", f"{mins_left}:{secs_left:02d}")

        st.subheader("Player profile")
        p1, p2, p3, p4 = st.columns(4)
        def format_shot_count(n: int) -> str:
            if n >= 1000:
                return f"{n / 1000:.1f}K"
            return str(n)

        p1.metric("Shots", format_shot_count(psummary["shots"]))
        p2.metric("FG%", f"{psummary['fg_pct'] * 100:.0f}%")
        p3.metric("3P%", f"{psummary['three_pct'] * 100:.0f}%")
        p4.metric("Rim FG%", f"{psummary['rim_pct'] * 100:.0f}%")

    with top_right:
        fig = draw_half_court(loc_x, loc_y, probability)
        st.pyplot(fig, width="stretch")

    st.divider()

    bottom_left, bottom_right = st.columns([1, 1])

    with bottom_left:
        st.subheader("Input summary")
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
                str(home_choice),
            ],
        })
        st.dataframe(summary, width="stretch", hide_index=True)

    with bottom_right:
        st.subheader("Why this prediction?")
        explanation = build_explanation(input_df, probability, league_avg)

        for title, detail in explanation:
            st.markdown(
                f"""
                <div class="info-card">
                    <b>{title}</b><br>
                    <span class="subtle">{detail}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with st.expander("Model notes"):
        st.write(
            """
            This prediction uses a trained XGBoost expected-field-goal model.
            Features include shot geometry, leakage-free player historical priors,
            team offensive context, opponent defensive context, shot profile interactions,
            player shot tendencies, and residual player skill features.

            The explanation section is a readable model-context summary. A future version
            can add true per-shot SHAP explanations.
            """
        )


if __name__ == "__main__":
    main()