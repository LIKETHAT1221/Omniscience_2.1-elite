import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import sqlite3

# ---------------------------
# Import custom modules
# ---------------------------
from backtester import Backtester
from line_movement import track_line_movement
from ta_engine import TechnicalAnalysisEngine
from betting_analyzer import BettingAnalyzer
from recommendations_engine import RecommendationsEngine
from odds_utils import american_to_prob, no_vig_prob

# ---------------------------
# Initialize engines
# ---------------------------
ta_engine = TechnicalAnalysisEngine()
betting_analyzer = BettingAnalyzer()
recommendations_engine = RecommendationsEngine()
backtester = Backtester()

# ---------------------------
# SQLite setup
# ---------------------------
conn = sqlite3.connect("omniscience.db")
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS analysis_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    raw_data TEXT,
    rec TEXT,
    conf TEXT
)
""")
conn.commit()

# ---------------------------
# Streamlit setup
# ---------------------------
st.set_page_config(page_title="Omniscience - Enhanced TA Engine", page_icon="📊", layout="wide")
st.title("Omniscience — Enhanced TA Engine (EV + AdaptiveMA + Forecasting)")

# ---------------------------
# Bankroll input
# ---------------------------
bankroll = st.number_input("Bankroll ($)", value=1000.0, min_value=1.0, step=100.0)

# ---------------------------
# Odds feed input
# ---------------------------
st.subheader("Odds Feed Input")
raw_data = st.text_area(
    "Paste odds data here (first line should be header)",
    height=300,
    placeholder="time 10/15 12:00PM\nLAC -3.5\n-110\nO 154.5\n-110\n-120 105\n\n"
)


# ---------------------------
# Clear button
# ---------------------------
if st.button("Clear Data"):
    raw_data = ""
    st.experimental_rerun()  # refreshes the page so the text area clears

# ---------------------------
# Parsing helpers
# ---------------------------
def is_header_line(line: str) -> bool:
    return line.lower().startswith("time")

def parse_blocks_strict(raw: str):
    lines = [l.strip() for l in raw.split("\n") if l.strip()]
    start = 1 if is_header_line(lines[0]) else 0
    rows = []

    for i in range(start, len(lines)-4, 5):
        try:
            L1, L2, L3, L4, L5 = lines[i:i+5]

            # Timestamp & team
            date_part, time_part, *team_tokens = L1.split()
            month, day = map(int, date_part.split("/"))
            hour, minute = map(int, time_part[:-2].split(":"))
            period = time_part[-2:].upper()
            if period == "PM" and hour < 12: hour += 12
            if period == "AM" and hour == 12: hour = 0
            time = datetime(2000, month, day, hour, minute)
            team = team_tokens[0] if team_tokens else None
            spread = float(team_tokens[1]) if len(team_tokens) > 1 else None
            spread_vig = float(L2)

            # Total
            total_side = L3[0].lower() if L3[0].lower() in ["o","u"] else None
            total = float(L3.split()[1])
            total_vig = float(L4)

            # Moneylines
            ml_tokens = L5.split()
            ml_away = float(ml_tokens[0]) if ml_tokens[0].lower() != "even" else 100
            ml_home = float(ml_tokens[1]) if ml_tokens[1].lower() != "even" else 100

            # Probabilities
            spread_prob = american_to_prob(spread_vig)
            spread_opposite_prob = 1 - spread_prob
            total_prob = american_to_prob(total_vig)
            total_opposite_prob = 1 - total_prob
            ml_no_vig = (american_to_prob(ml_away), american_to_prob(ml_home))

            # Add delta/momentum placeholders
            row = {
                "time": time, "team": team, "spread": spread, "spread_vig": spread_vig,
                "spread_prob": spread_prob, "spread_opposite_prob": spread_opposite_prob,
                "total": total, "total_side": total_side, "total_vig": total_vig,
                "total_prob": total_prob, "total_opposite_prob": total_opposite_prob,
                "ml_away": ml_away, "ml_home": ml_home, "ml_no_vig": ml_no_vig,
                "spread_delta": 0, "spread_momentum": 0, "spread_momentum_smooth": 0,
                "total_delta": 0, "total_momentum": 0, "total_momentum_smooth": 0
            }
            rows.append(row)
        except Exception as e:
            st.error(f"Error parsing block at line {i}: {e}")
            continue
    return rows

# ---------------------------
# Analyze Button
# ---------------------------
if st.button("Analyze"):
    if not raw_data.strip():
        st.error("Please paste odds data first.")
    else:
        parsed_data = parse_blocks_strict(raw_data)

        # Track line movement
        parsed_data = track_line_movement(parsed_data, "spread")
        parsed_data = track_line_movement(parsed_data, "total")

        # Apply TA engine
        ta_results = ta_engine.analyze_all(parsed_data)

        # Betting analysis
        betting_results = betting_analyzer.analyze_all(parsed_data)

        # Recommendations
        recommendations = recommendations_engine.generate_recommendations(
            ta_results, betting_results, bankroll
        )

        # Store results
        cursor.execute(
            "INSERT INTO analysis_results (timestamp, raw_data, rec, conf) VALUES (?, ?, ?, ?)",
            (datetime.now().isoformat(), raw_data, recommendations['rec'], recommendations['conf'])
        )
        conn.commit()

        # Display results
        st.subheader("Analysis Results")
        st.markdown(recommendations['html'], unsafe_allow_html=True)

        # Display parsed preview with delta/momentum
        st.subheader("Parsed Data with Delta / Momentum")
        preview_df = pd.DataFrame([
            {
                'Time': row['time'].strftime('%m/%d %H:%M'),
                'Team': row['team'],
                'Spread': row['spread'],
                'Spread Δ': row['spread_delta'],
                'Spread Momentum': row['spread_momentum'],
                'Spread Momentum (Smooth)': row['spread_momentum_smooth'],
                'Total': row['total'],
                'Total Δ': row['total_delta'],
                'Total Momentum': row['total_momentum'],
                'Total Momentum (Smooth)': row['total_momentum_smooth'],
                'Spread Prob': row['spread_prob'],
                'Spread Opposite Prob': row['spread_opposite_prob'],
                'Total Prob': row['total_prob'],
                'Total Opposite Prob': row['total_opposite_prob'],
                'Away ML': row['ml_away'],
                'Home ML': row['ml_home']
            } for row in parsed_data
        ])
        st.dataframe(preview_df, use_container_width=True)

# ---------------------------
# Backtesting Button
# ---------------------------
if st.button("Run Backtest"):
    if not raw_data.strip():
        st.error("Please paste odds data first.")
    else:
        parsed_data = parse_blocks_strict(raw_data)
        backtest_result = backtester.run_backtest(parsed_data, bankroll)
        st.subheader("Backtest Results")
        st.markdown(backtest_result['html'], unsafe_allow_html=True)
