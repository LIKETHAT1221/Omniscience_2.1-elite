import pandas as pd
import numpy as np

class RecommendationsEngine:
    def __init__(self):
        pass

    def generate_recommendations(self, ta_results, betting_results, bankroll):
        """
        Combine TA and betting analysis into actionable recommendations.
        Returns dict with:
        - rec: text recommendation
        - conf: confidence score (0-100)
        - html: formatted HTML for Streamlit
        """
        rec_list = []
        conf_list = []

        html = "<h4>Recommendations</h4><ul>"

        for i, (ta_row, bet_row) in enumerate(zip(ta_results, betting_results)):
            # ---------------------------
            # Base EV weighting
            # ---------------------------
            combined_ev = ta_row['combined_ev']  # already includes spread_ev + total_ev + ml_ev

            # ---------------------------
            # TA confirmations
            # ---------------------------
            confirmations = 0

            # AdaptiveMA trend confirmation
            if ta_row['spread'] > ta_row['adaptive_ma_spread']:
                confirmations += 1
            if ta_row['total'] > ta_row['adaptive_ma_total']:
                confirmations += 1

            # Momentum confirmation
            if ta_row['mom_a_spread'] > 0:
                confirmations += 1
            if ta_row['mom_a_total'] > 0:
                confirmations += 1

            # Steam / RLM / FLM signals
            confirmations += ta_row['steam_signal']
            confirmations += ta_row['rlm_signal']
            confirmations += ta_row['flm_signal']

            # Z-score extremes (mean reversion signals)
            if abs(ta_row['spread_z']) > 1.5:
                confirmations += 1
            if abs(ta_row['total_z']) > 1.5:
                confirmations += 1

            # Fibonacci retracement confirmation (price near retracement)
            spread_close_to_fib = abs(ta_row['spread'] - ta_row['fib_retracement_spread']) / (ta_row['spread'] + 1e-6)
            total_close_to_fib = abs(ta_row['total'] - ta_row['fib_retracement_total']) / (ta_row['total'] + 1e-6)
            if spread_close_to_fib < 0.02:  # 2% tolerance
                confirmations += 1
            if total_close_to_fib < 0.02:
                confirmations += 1

            # ---------------------------
            # Confidence calculation
            # ---------------------------
            # Base: scale combined EV to 0-1
            base_conf = np.clip(combined_ev * 100, -100, 100)
            # TA confirmations weight (10 points each)
            conf_score = base_conf + confirmations * 10
            # Clip to 0-100 for display
            conf_score = np.clip(conf_score, 0, 100)

            # ---------------------------
            # Recommendation logic
            # ---------------------------
            if conf_score > 60:
                rec_text = f"Strong Bet on {ta_row['team']}"
            elif conf_score > 40:
                rec_text = f"Moderate Bet / Caution on {ta_row['team']}"
            else:
                rec_text = f"Avoid {ta_row['team']}"

            # ---------------------------
            # Append HTML
            # ---------------------------
            html += (
                f"<li><strong>{ta_row['team']}</strong>: {rec_text}, "
                f"Confidence: {conf_score:.1f}%</li>"
            )

            rec_list.append(rec_text)
            conf_list.append(conf_score)

        html += "</ul>"

        # Return final dict
        return {
            'rec': ', '.join(rec_list),
            'conf': ', '.join([f"{c:.1f}" for c in conf_list]),
            'html': html
        }
