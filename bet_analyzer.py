import numpy as np

class BettingAnalyzer:
    def __init__(self):
        pass

    def analyze_all(self, parsed_data):
        """
        Compute spread, total, moneyline analysis for each row.
        Returns list of dicts same length as parsed_data.
        """
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
