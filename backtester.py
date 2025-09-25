import pandas as pd
import numpy as np

class Backtester:
    def __init__(self):
        pass

    def run_backtest(self, parsed_data, bankroll):
        """
        Run a simple backtest using combined EV and Kelly fraction.
        Returns dict with HTML summary.
        """
        df = pd.DataFrame(parsed_data)
        if 'combined_ev' not in df.columns or 'kelly_fraction' not in df.columns:
            # placeholder calculation if not present
            df['combined_ev'] = 0.05
            df['kelly_fraction'] = 0.05

        results = []
        bankroll_progress = bankroll

        for i, row in df.iterrows():
            bet_size = bankroll_progress * row.get('kelly_fraction', 0.05)
            # Simulate win/loss: if EV positive, win; else lose (simplified)
            if row['combined_ev'] > 0:
                pnl = bet_size * row['combined_ev']
            else:
                pnl = -bet_size * abs(row['combined_ev'])
            bankroll_progress += pnl
            results.append({
                'team': row.get('team'),
                'bet_size': bet_size,
                'pnl': pnl,
                'bankroll': bankroll_progress
            })

        # Generate HTML
        html = "<h4>Backtest Results</h4><table><tr><th>Team</th><th>Bet Size</th><th>PnL</th><th>Bankroll</th></tr>"
        for r in results:
            html += f"<tr><td>{r['team']}</td><td>{r['bet_size']:.2f}</td><td>{r['pnl']:.2f}</td><td>{r['bankroll']:.2f}</td></tr>"
        html += "</table>"

        return {'html': html, 'final_bankroll': bankroll_progress}
