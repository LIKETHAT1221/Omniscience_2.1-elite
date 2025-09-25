import pandas as pd
import numpy as np

class TechnicalAnalysisEngine:
    def __init__(self, smooth_window=5, fib_window=10, steam_threshold=0.05):
        self.smooth_window = smooth_window
        self.fib_window = fib_window
        self.steam_threshold = steam_threshold  # fraction change to trigger steam

    def analyze_all(self, parsed_data):
        """
        Run full operational TA analysis over the entire parsed_data feed.
        Returns list of dicts with complete TA signals.
        """
        results = []

        df = pd.DataFrame(parsed_data)

        # ---------------------------
        # AdaptiveMA (primary driver)
        # ---------------------------
        df['adaptive_ma_spread'] = df['spread'].ewm(span=self.smooth_window, adjust=False).mean()
        df['adaptive_ma_total'] = df['total'].ewm(span=self.smooth_window, adjust=False).mean()

        # ---------------------------
        # EV (primary driver)
        # ---------------------------
        df['spread_ev'] = df['spread_prob'] - 0.5
        df['total_ev'] = df['total_prob'] - 0.5
        df['ml_ev'] = df['ml_no_vig'].apply(lambda x: max(x) - 0.5)
        df['combined_ev'] = df['spread_ev'] + df['total_ev'] + df['ml_ev']

        # ---------------------------
        # SMA / EMA
        # ---------------------------
        df['sma_spread'] = df['spread'].rolling(window=self.smooth_window, min_periods=1).mean()
        df['ema_spread'] = df['spread'].ewm(span=self.smooth_window, adjust=False).mean()
        df['sma_total'] = df['total'].rolling(window=self.smooth_window, min_periods=1).mean()
        df['ema_total'] = df['total'].ewm(span=self.smooth_window, adjust=False).mean()

        # ---------------------------
        # Bollinger Bands
        # ---------------------------
        df['spread_std'] = df['spread'].rolling(window=self.smooth_window, min_periods=1).std().fillna(0)
        df['spread_upper'] = df['sma_spread'] + 2 * df['spread_std']
        df['spread_lower'] = df['sma_spread'] - 2 * df['spread_std']

        df['total_std'] = df['total'].rolling(window=self.smooth_window, min_periods=1).std().fillna(0)
        df['total_upper'] = df['sma_total'] + 2 * df['total_std']
        df['total_lower'] = df['sma_total'] - 2 * df['total_std']

        # ---------------------------
        # Z-score
        # ---------------------------
        df['spread_z'] = (df['spread'] - df['sma_spread']) / (df['spread_std'] + 1e-6)
        df['total_z'] = (df['total'] - df['sma_total']) / (df['total_std'] + 1e-6)

        # ---------------------------
        # ATR (average true range) via delta
        # ---------------------------
        df['atr_spread'] = df['spread_delta'].abs().rolling(window=self.smooth_window, min_periods=1).mean()
        df['atr_total'] = df['total_delta'].abs().rolling(window=self.smooth_window, min_periods=1).mean()

        # ---------------------------
        # MOM_A / MOM_V
        # ---------------------------
        df['mom_a_spread'] = df['spread_mom_a']
        df['mom_v_spread'] = df['spread_mom_v']
        df['mom_a_total'] = df['total_mom_a']
        df['mom_v_total'] = df['total_mom_v']

        # ---------------------------
        # Fibonacci Retracement / Extension
        # ---------------------------
        df['fib_high_spread'] = df['spread'].rolling(window=self.fib_window, min_periods=1).max()
        df['fib_low_spread'] = df['spread'].rolling(window=self.fib_window, min_periods=1).min()
        df['fib_retracement_spread'] = df['fib_high_spread'] - 0.618*(df['fib_high_spread'] - df['fib_low_spread'])
        df['fib_extension_spread'] = df['fib_high_spread'] + 0.618*(df['fib_high_spread'] - df['fib_low_spread'])

        df['fib_high_total'] = df['total'].rolling(window=self.fib_window, min_periods=1).max()
        df['fib_low_total'] = df['total'].rolling(window=self.fib_window, min_periods=1).min()
        df['fib_retracement_total'] = df['fib_high_total'] - 0.618*(df['fib_high_total'] - df['fib_low_total'])
        df['fib_extension_total'] = df['fib_high_total'] + 0.618*(df['fib_high_total'] - df['fib_low_total'])

        # ---------------------------
        # Steam Detection
        # ---------------------------
        df['steam_signal'] = 0
        for i in range(1, len(df)):
            # % change in spread/total
            spread_change = abs(df.loc[i, 'spread'] - df.loc[i-1, 'spread']) / (df.loc[i-1, 'spread'] + 1e-6)
            total_change = abs(df.loc[i, 'total'] - df.loc[i-1, 'total']) / (df.loc[i-1, 'total'] + 1e-6)
            if spread_change >= self.steam_threshold or total_change >= self.steam_threshold:
                df.loc[i, 'steam_signal'] = 1 if df.loc[i, 'spread'] - df.loc[i-1, 'spread'] > 0 else -1

        # ---------------------------
        # RLM / FLM Detection
        # ---------------------------
        df['rlm_signal'] = 0
        df['flm_signal'] = 0
        for i in range(1, len(df)):
            # If line moves opposite to EV → RLM
            if np.sign(df.loc[i, 'spread_delta']) != np.sign(df.loc[i, 'spread_ev']):
                df.loc[i, 'rlm_signal'] = 1
            # If line moves in same direction as EV → FLM
            if np.sign(df.loc[i, 'spread_delta']) == np.sign(df.loc[i, 'spread_ev']):
                df.loc[i, 'flm_signal'] = 1

        # ---------------------------
        # ROI / Kelly placeholder (can integrate bankroll later)
        # ---------------------------
        df['roi_spread'] = df['combined_ev']
        df['kelly_fraction'] = df['combined_ev']

        # ---------------------------
        # Compile results row by row
        # ---------------------------
        for i, row in df.iterrows():
            results.append({
                'team': row['team'],
                'spread': row['spread'],
                'total': row['total'],
                'spread_ev': row['spread_ev'],
                'total_ev': row['total_ev'],
                'ml_ev': row['ml_ev'],
                'combined_ev': row['combined_ev'],
                'adaptive_ma_spread': row['adaptive_ma_spread'],
                'adaptive_ma_total': row['adaptive_ma_total'],
                'sma_spread': row['sma_spread'],
                'ema_spread': row['ema_spread'],
                'sma_total': row['sma_total'],
                'ema_total': row['ema_total'],
                'spread_upper': row['spread_upper'],
                'spread_lower': row['spread_lower'],
                'total_upper': row['total_upper'],
                'total_lower': row['total_lower'],
                'spread_z': row['spread_z'],
                'total_z': row['total_z'],
                'atr_spread': row['atr_spread'],
                'atr_total': row['atr_total'],
                'mom_a_spread': row['mom_a_spread'],
                'mom_v_spread': row['mom_v_spread'],
                'mom_a_total': row['mom_a_total'],
                'mom_v_total': row['mom_v_total'],
                'fib_retracement_spread': row['fib_retracement_spread'],
                'fib_extension_spread': row['fib_extension_spread'],
                'fib_retracement_total': row['fib_retracement_total'],
                'fib_extension_total': row['fib_extension_total'],
                'steam_signal': row['steam_signal'],
                'rlm_signal': row['rlm_signal'],
                'flm_signal': row['flm_signal'],
                'roi_spread': row['roi_spread'],
                'kelly_fraction': row['kelly_fraction']
            })

        return results
