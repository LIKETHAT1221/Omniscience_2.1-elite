# omniscience.py - Fully Merged Version
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import sqlite3

# ---------------------------
# Odds Utils
# ---------------------------
def american_to_prob(american_odds):
    if american_odds > 0:
        return 100 / (american_odds + 100)
    else:
        return -american_odds / (-american_odds + 100)

def no_vig_prob(prob):
    return prob / sum(prob) if isinstance(prob, (list, tuple)) else prob

def american_to_decimal(american_odds):
    if american_odds > 0:
        return 1 + (american_odds / 100)
    else:
        return 1 + (100 / abs(american_odds))

def decimal_to_american(decimal_odds):
    if decimal_odds >= 2:
        return (decimal_odds - 1) * 100
    else:
        return -100 / (decimal_odds - 1)

# ---------------------------
# Track Line Movement
# ---------------------------
def track_line_movement(parsed_data, field, smooth_window=5):
    df = pd.DataFrame(parsed_data)
    if field not in df.columns:
        raise ValueError(f"{field} not in data")

    df[f'{field}_delta'] = df[field].diff().fillna(0)
    df[f'{field}_momentum'] = df[f'{field}_delta'].diff().fillna(0)
    df[f'{field}_momentum_smooth'] = df[f'{field}_momentum'].rolling(window=smooth_window, min_periods=1).mean()
    df[f'{field}_mom_a'] = df[f'{field}_momentum'].rolling(window=smooth_window, min_periods=1).mean()
    df[f'{field}_mom_v'] = df[f'{field}_momentum'].rolling(window=smooth_window, min_periods=1).std().fillna(0)

    return df.to_dict(orient='records')

# ---------------------------
# Betting Analyzer
# ---------------------------
class BettingAnalyzer:
    def __init__(self):
        pass

    def analyze_all(self, parsed_data):
        results = []
        for row in parsed_data:
            spread_edge = row['spread_no_vig'] - 0.5
            total_edge = row['total_no_vig'] - 0.5
            ml_edge = max(row['ml_no_vig']) - 0.5 if row.get('ml_no_vig') else 0

            results.append({
                'spread_edge': spread_edge,
                'total_edge': total_edge,
                'ml_edge': ml_edge,
                'combined_edge': spread_edge + total_edge + ml_edge
            })
        return results

# ---------------------------
# Backtester
# ---------------------------
class Backtester:
    def __init__(self):
        self.debug = True

    def run_backtest(self, parsed_data, bankroll):
        results_summary = []

        for row in parsed_data:
            spread_ev = row['spread_no_vig'] - 0.5
            total_ev = row['total_no_vig'] - 0.5
            ml_ev = max(row['ml_no_vig']) - 0.5 if row.get('ml_no_vig') else 0
            combined_ev = spread_ev + total_ev + ml_ev

            if combined_ev > 0.05:
                recommendation = f"Bet on {row['team']}"
            elif combined_ev < -0.05:
                recommendation = f"Avoid {row['team']}"
            else:
                recommendation = "Hold"

            results_summary.append({
                'team': row['team'],
                'spread_prob': row['spread_no_vig'],
                'total_prob': row['total_no_vig'],
                'ml_away_prob': row['ml_no_vig'][0] if row.get('ml_no_vig') else None,
                'ml_home_prob': row['ml_no_vig'][1] if row.get('ml_no_vig') else None,
                'spread_delta': row.get('spread_delta'),
                'spread_momentum': row.get('spread_momentum'),
                'total_delta': row.get('total_delta'),
                'total_momentum': row.get('total_momentum'),
                'recommendation': recommendation
            })

            if self.debug:
                print(f"[DEBUG] {row['team']} - Spread EV: {spread_ev:.3f}, "
                      f"Total EV: {total_ev:.3f}, ML EV: {ml_ev:.3f}, Combined EV: {combined_ev:.3f}, "
                      f"Recommendation: {recommendation}")

        html = "<h4>Backtest Results</h4><ul>"
        for r in results_summary:
            html += (
                f"<li><strong>{r['team']}</strong>: Spread Δ={r['spread_delta']}, "
                f"Spread Momentum={r['spread_momentum']}, Total Δ={r['total_delta']}, "
                f"Total Momentum={r['total_momentum']}, "
                f"Spread Prob={r['spread_prob']:.2f}, Total Prob={r['total_prob']:.2f}, "
                f"Away ML Prob={r['ml_away_prob']}, Home ML Prob={r['ml_home_prob']}, "
                f"Recommendation={r['recommendation']}</li>"
            )
        html += "</ul>"
        return {"html": html}

# ---------------------------
# Parsing
# ---------------------------
def is_header_line(line: str) -> bool:
    return line.lower().startswith("time")

def parse_blocks_strict(raw: str):
    lines = [l.strip() for l in raw.split("\n") if l.strip()]
    start = 1 if is_header_line(lines[0]) else 0
    rows = []

    for i in range(start, len(lines)-4, 5):
        block = lines[i:i+5]
        try:
            L1, L2, L3, L4, L5 = block
            # Timestamp
            date_part, time_part, *team_tokens = L1.split()
            month, day = map(int, date_part.split("/"))
            hour, minute = map(int, time_part[:-2].split(":"))
            period = time_part[-2:].upper()
            if period == "PM" and hour < 12: hour += 12
            if period == "AM" and hour == 12: hour = 0
            time = datetime(2000, month, day, hour, minute)
            # Team & spread
            team = team_tokens[0] if team_tokens else None
            spread = float(team_tokens[1]) if len(team_tokens) > 1 else None
            spread_vig = float(L2)
            total_side = L3[0].lower() if L3[0].lower() in ["o","u"] else None
            total = float(L3.split()[1])
            total_vig = float(L4)
            # Moneylines
            ml_tokens = L5.split()
            ml_away, ml_home = None, None
            for t in ml_tokens:
                if t.lower() == "even": t = "+100"
                if ml_away is None: ml_away = float(t)
                else: ml_home = float(t)
            # No-vig probabilities
            spread_no_vig = american_to_prob(spread_vig)
            total_no_vig = american_to_prob(total_vig)
            ml_no_vig = (american_to_prob(ml_away), american_to_prob(ml_home)) if ml_away and ml_home else None

            rows.append({
                "time": time, "team": team, "spread": spread, "spread_vig": spread_vig,
                "total": total, "total_side": total_side, "total_vig": total_vig,
                "ml_away": ml_away, "ml_home": ml_home,
                "spread_no_vig": spread_no_vig, "total_no_vig": total_no_vig, "ml_no_vig": ml_no_vig
            })
        except Exception as e:
            st.error(f"Error parsing block at line {i}: {e}")
            continue
    return rows

# ---------------------------
# Initialize engines
# ---------------------------
ta_engine = None  # Placeholder if you don’t merge TA engine yet
betting_analyzer = BettingAnalyzer()
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
# Streamlit App
# ---------------------------
st.set_page_config(page_title="Omniscience - Enhanced TA Engine", page_icon="📊", layout="wide")
st.title("Omniscience — Enhanced TA Engine (EV + Backtesting)")

# Bankroll input
bankroll = st.number_input("Bankroll ($)", value=1000.0, min_value=1.0, step=100.0)

# Odds feed input
st.subheader("Odds Feed Input")
raw_data = st.text_area(
    "Paste odds data here (first line should be header)",
    height=300,
    placeholder="time 10/15 12:00PM\nLAC -3.5\n-110\nO 154.5\n-110\n-120 105\n\n"
)

# Clear button
if st.button("Clear Odds Data"):
    st.experimental_rerun()

# ---------------------------
# Analyze Button
# ---------------------------
if st.button("Analyze"):
    if not raw_data.strip():
        st.error("Please paste odds data first.")
    else:
        parsed_data = parse_blocks_strict(raw_data)
        parsed_data = track_line_movement(parsed_data, "spread", smooth_window=5)
        parsed_data = track_line_movement(parsed_data, "total", smooth_window=5)

        # Run betting analysis
        betting_results = betting_analyzer.analyze_all(parsed_data)

        # Store results (dummy placeholders for rec/conf)
        rec = "See betting edges above"
        conf = "N/A"
        cursor.execute(
            "INSERT INTO analysis_results (timestamp, raw_data, rec, conf) VALUES (?, ?, ?, ?)",
            (datetime.now().isoformat(), raw_data, rec, conf)
        )
        conn.commit()

        # Display parsed preview with momentum
        st.subheader("Parsed Data with Line Movement")
        preview_df = pd.DataFrame(parsed_data)
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
