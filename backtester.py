class Backtester:
    def __init__(self):
        # You can initialize historical data or config here if needed
        self.debug = True  # set to True to print debug info

    def run_backtest(self, parsed_data, bankroll):
        """
        Run a simple backtest using parsed_data.
        Uses spread, total, and moneyline probabilities to simulate results.
        """

        # Store results
        results_summary = []

        for row in parsed_data:
            # Basic mock bet simulation for demonstration
            spread_ev = row['spread_prob'] - 0.5
            total_ev = row['total_prob'] - 0.5
            ml_ev = max(row['ml_no_vig']) - 0.5

            # Combine EVs as a simple metric
            combined_ev = spread_ev + total_ev + ml_ev

            # Generate recommendation for backtest report
            if combined_ev > 0.05:
                recommendation = f"Bet on {row['team']}"
            elif combined_ev < -0.05:
                recommendation = f"Avoid {row['team']}"
            else:
                recommendation = "Hold"

            # Append to summary
            results_summary.append({
                'team': row['team'],
                'spread_prob': row['spread_prob'],
                'total_prob': row['total_prob'],
                'ml_away_prob': row['ml_no_vig'][0],
                'ml_home_prob': row['ml_no_vig'][1],
                'spread_delta': row['spread_delta'],
                'spread_momentum': row['spread_momentum'],
                'total_delta': row['total_delta'],
                'total_momentum': row['total_momentum'],
                'recommendation': recommendation
            })

            if self.debug:
                print(f"[DEBUG] {row['team']} - Spread EV: {spread_ev:.3f}, "
                      f"Total EV: {total_ev:.3f}, ML EV: {ml_ev:.3f}, Combined EV: {combined_ev:.3f}, "
                      f"Recommendation: {recommendation}")

        # Build HTML report for Streamlit
        html = "<h4>Backtest Results</h4><ul>"
        for r in results_summary:
            html += (
                f"<li><strong>{r['team']}</strong>: Spread Δ={r['spread_delta']}, "
                f"Spread Momentum={r['spread_momentum']}, Total Δ={r['total_delta']}, "
                f"Total Momentum={r['total_momentum']}, "
                f"Spread Prob={r['spread_prob']:.2f}, Total Prob={r['total_prob']:.2f}, "
                f"Away ML Prob={r['ml_away_prob']:.2f}, Home ML Prob={r['ml_home_prob']:.2f}, "
                f"Recommendation={r['recommendation']}</li>"
            )
        html += "</ul>"

        return {"html": html}
