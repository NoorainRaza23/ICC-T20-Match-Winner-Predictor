import streamlit as st 
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import requests
from dotenv import load_dotenv
import os

# Load environment variables from .env
load_dotenv()
API_KEY = os.getenv("CRICAPI_KEY")

# Load model
pipe = pickle.load(open('pipe.pkl', 'rb'))

# Set layout
st.set_page_config(page_title="T20 World Cup Predictor", layout="wide")

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["🏏 Match-wise Predictor", "🏆 Final Prediction Result"])

# ===========================
# Final Winner Page
# ===========================
if page == "🏆 Final Prediction Result":
    st.title("🏆 Final World Cup Winner Prediction (Combined Model)")
    try:
        final_results = pd.read_excel("final_combined_results.xlsx")
        final_results = final_results.rename(columns={"index": "Team", "win_percent": "Win %"})

        final_results.reset_index(drop=True, inplace=True)
        final_results.index += 1

        winner_row = final_results.loc[final_results["Win %"].idxmax()]
        team_name = winner_row['Team'].strip()

        st.success(f"🏆 Predicted Winner:  **{team_name}**  with **{winner_row['Win %']:.2f}%** win chance!")
        st.dataframe(final_results)
    except Exception as e:
        st.error(f"❌ Could not load final results: {e}")

# ===========================
# Match-wise Predictor Page
# ===========================
elif page == "🏏 Match-wise Predictor":
    st.image("https://www.shutterstock.com/image-vector/indian-cricket-logo-flag-ribbon-600nw-2395209491.jpg", width=400)
    st.markdown("<h1 style='text-align: center;'>ICC T20 Match Winner Predictor</h1>", unsafe_allow_html=True)

    teams = sorted([
        'West Indies', 'Netherlands', 'United States of America', 'Bangladesh',
        'Pakistan', 'Australia', 'Ireland', 'Scotland', 'Afghanistan', 'New Zealand',
        'England', 'Uganda', 'South Africa', 'Namibia', 'India', 'Nepal', 'Oman'
    ])
    cities = ['Bridgetown', 'Dallas', 'North Sound', 'New York', 'Providence',
              'Gros Islet', 'Kingstown', 'Tarouba', 'Lauderhill']

    col1, col2 = st.columns(2)
    with col1:
        batting_team = st.selectbox('🏏 Batting Team', teams)
    with col2:
        bowling_team = st.selectbox('🎯 Bowling Team', teams)
    selected_city = st.selectbox('📍 Match Venue', sorted(cities))

    col3, col4, col5 = st.columns(3)
    with col3:
        target = st.number_input('🎯 Target Score', min_value=1)
    with col4:
        score = st.number_input('🏏 Current Score', min_value=0)
    with col5:
        wickets = st.number_input('❌ Wickets Lost', min_value=0, max_value=9)

    overs = st.slider('🕐 Overs Completed', min_value=0, max_value=20, step=1, value=0)

    if st.button('🚀 Predict Win Probability'):
        runs_left = target - score
        balls_left = 120 - (overs * 6)
        remaining_wickets = 10 - wickets
        crr = score / overs if overs > 0 else 0
        rrr = (runs_left * 6 / balls_left) if balls_left > 0 else 0

        input_df = pd.DataFrame({
            'batting_team': [batting_team],
            'bowling_team': [bowling_team],
            'city': [selected_city],
            'runs_left': [runs_left],
            'balls_left': [balls_left],
            'wickets': [remaining_wickets],
            'total_runs_x': [target],
            'crr': [crr],
            'rrr': [rrr]
        })

        result = pipe.predict_proba(input_df)
        loss = round(result[0][0] * 100)
        win = round(result[0][1] * 100)

        st.markdown("## 📊 Prediction Result")
        st.success(f"🏆 {batting_team} Win Chance: **{win}%**")
        st.warning(f"🛡️ {bowling_team} Win Chance: **{loss}%**")
        st.progress(win / 100)

        st.markdown("### 📈 CRR vs RRR Comparison")
        fig, ax = plt.subplots()
        ax.bar(['Current RR', 'Required RR'], [crr, rrr], color=['green', 'red'])
        ax.set_ylabel("Runs per Over")
        st.pyplot(fig)

        st.markdown("### 📄 Match Summary")
        st.write(f"🏏 **Current Score**: {score}/{wickets}")
        st.write(f"🎯 **Target**: {target}")
        st.write(f"⚖️ **Runs Required**: {runs_left} in {balls_left} balls")
        st.write(f"🔥 **Current Run Rate (CRR)**: {round(crr, 2)}")
        st.write(f"🚨 **Required Run Rate (RRR)**: {round(rrr, 2)}")

        st.markdown("### ⚡ Momentum Indicator")
        if win > 70:
            st.success("🔵 Momentum with Batting Team!")
        elif loss > 70:
            st.error("🔴 Bowling Team Dominating!")
        else:
            st.info("🟡 Match is evenly poised!")

        st.markdown("### 🧠 Strategy Tips")
        if rrr > 10:
            st.warning("🔺 Try to target weaker bowlers!")
        elif remaining_wickets <= 3:
            st.warning("⚠️ Be cautious, wickets in hand are crucial now!")
        else:
            st.success("✅ You can pace the innings calmly with this RRR.")

    # Live Matches Section
    st.header("🟢 Live Matches")

    def get_live_matches():
        url = f"https://api.cricapi.com/v1/currentMatches?apikey={API_KEY}&offset=0"
        try:
            res = requests.get(url)
            if res.status_code == 200:
                data = res.json()
                if data['status'] != "success":
                    return "❌ API Error", []
                return None, data.get("data", [])
            else:
                return f"❌ Error: {res.status_code}", []
        except:
            return "❌ Could not reach API", []

    error, live_matches = get_live_matches()

    if error:
        st.error(error)
    elif not live_matches:
        st.info("No live matches at the moment.")
    else:
        for match in live_matches:
            try:
                st.subheader(match.get('name', 'Match Name N/A'))
                st.write(f"📍 Venue: {match.get('venue', 'N/A')}")
                st.write(f"🕒 Date: {match.get('date', 'N/A')}")
                st.write(f"🔄 Status: {match.get('status', 'N/A')}")
                teams = match.get('teams', ['Team A', 'Team B'])
                st.write(f"🏏 Match: {teams[0]} vs {teams[1]}")
                st.divider()
            except Exception as e:
                st.warning(f"Could not display match due to: {e}")