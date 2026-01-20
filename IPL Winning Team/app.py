import streamlit as st
import pandas as pd
import pickle
import altair as alt

st.title("IPL Winning Team Probability Predictor")

# Load trained model
model = pickle.load(open('ipl_win_model.pkl', 'rb'))

batting_teams = [
    'Chennai Super Kings', 'Delhi Daredevils', 'Mumbai Indians', 'Kolkata Knight Riders',
    'Kings XI Punjab', 'Rajasthan Royals', 'Royal Challengers Bangalore', 'Sunrisers Hyderabad'
]
bowling_teams = batting_teams.copy()

venues = ['Mumbai', 'Chennai', 'Jaipur', 'Hyderabad', 'Bangalore', 'Punjab', 'Delhi', 'Kolkata']

# For year selection, add a range or list of IPL years as needed
years = [2020, 2021, 2022, 2023, 2024, 2025]  # Update according to your available data/model training

# Layout inputs in two columns
col1, col2 = st.columns(2)
with col1:
    batting_team = st.selectbox('Batting Team', batting_teams)
    bowling_team = st.selectbox('Bowling Team', bowling_teams)
    venue = st.selectbox('Venue/City', venues)
    year = st.selectbox('Year', years)

with col2:
    target = st.number_input("Target Score", min_value=1)
    score = st.number_input("Current Score", min_value=0)
    overs = st.number_input('Overs Completed', min_value=0.0, max_value=20.0, format="%.1f")
    wickets = st.number_input("Wickets Lost", min_value=0, max_value=10)

if st.button("Predict"):
    runs_left = target - score
    balls_left = 120 - (int(overs * 6))
    wickets_left = 10 - wickets
    crr = score / overs if overs > 0 else 0
    rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0

    input_df = pd.DataFrame({
        'batting_team': [batting_team],
        'bowling_team': [bowling_team],
        'venue': [venue],
        'runs_left': [runs_left],
        'balls_left': [balls_left],
        'wickets_left': [wickets_left],
        'crr': [crr],
        'rrr': [rrr],
        'year': [year]  # Added year to the input features
    })

    pred_proba = model.predict_proba(input_df)
    batting_prob = pred_proba[0][1] * 100
    bowling_prob = pred_proba[0][0] * 100

    st.subheader("Winning Probabilities")
    col3, col4 = st.columns(2)
    col3.metric(label=batting_team, value=f"{batting_prob:.2f}%")
    col4.metric(label=bowling_team, value=f"{bowling_prob:.2f}%")

    import matplotlib.pyplot as plt

    def circular_progress(ax, percent, label, color):
        ax.pie([percent, 100 - percent],
               startangle=90,
               colors=[color, 'lightgray'],
               wedgeprops=dict(width=0.3))
        ax.text(0, 0, f"{percent:.1f}%", ha='center', va='center', fontsize=14, fontweight='bold')
        ax.set_title(label)

    fig, axs = plt.subplots(1, 2, figsize=(6, 3))
    circular_progress(axs[0], batting_prob, batting_team, 'orange')
    circular_progress(axs[1], bowling_prob, bowling_team, 'blue')
    st.pyplot(fig)

    chart_data = pd.DataFrame({
        'Team': [batting_team, bowling_team],
        'Winning Probability (%)': [batting_prob, bowling_prob]
    })

    chart = (
        alt.Chart(chart_data)
        .mark_arc()
        .encode(
            theta=alt.Theta(field="Winning Probability (%)", type="quantitative"),
            color=alt.Color(field="Team", type="nominal"),
            tooltip=['Team', 'Winning Probability (%)']
        )
        .properties(
            width=400,
            height=300,
            title='Winning Probability Distribution'
        )
    )
    st.altair_chart(chart, use_container_width=True)
