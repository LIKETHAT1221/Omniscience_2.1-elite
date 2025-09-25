import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ---------------------------
# CORE HELPER FUNCTIONS
# ---------------------------

def american_to_prob(odds):
    """Convert American odds to probability"""
    if isinstance(odds, str) and odds.lower() == 'even':
        return 0.5
    odds = float(odds)
    if odds > 0:
        return 100 / (odds + 100)
    return abs(odds) / (abs(odds) + 100)

def no_vig_prob(odds1, odds2):
    """Remove vig from probabilities"""
    p1 = american_to_prob(odds1)
    p2 = american_to_prob(odds2)
    total = p1 + p2
    return (p1/total, p2/total)

def calculateEV(prob, odds):
    """Calculate Expected Value"""
    if isinstance(odds, str) and odds.lower() == 'even':
        odds = 100
    dec_odds = 1 + (odds / 100) if odds > 0 else 1 + (100 / abs(odds))
    return prob * (dec_odds - 1) - (1 - prob)

def calculateKelly(prob, odds):
    """Calculate Kelly Criterion bet size"""
    if isinstance(odds, str) and odds.lower() == 'even':
        odds = 100
    dec_odds = 1 + (odds / 100) if odds > 0 else 1 + (100 / abs(odds))
    B = dec_odds - 1
    P = prob
    Q = 1 - P
    return max((B*P - Q)/B, 0)

# ---------------------------
# TECHNICAL ANALYSIS STACK
# ---------------------------

class TAStack:
    def __init__(self, window=20):
        self.window = window
    
    def sma(self, data):
        """Simple Moving Average"""
        return pd.Series(data).rolling(window=self.window).mean()
    
    def ema(self, data, alpha=0.1):
        """Exponential Moving Average"""
        return pd.Series(data).ewm(alpha=alpha).mean()
    
    def adaptive_ma(self, data, volatility_window=10):
        """Adaptive Moving Average based on volatility"""
        returns = np.diff(data)
        volatility = pd.Series(returns).rolling(volatility_window).std()
        # Higher volatility = shorter MA, lower volatility = longer MA
        adaptive_alpha = 2 / (1 + np.exp(-volatility * 10)) - 1
        adaptive_alpha = np.clip(adaptive_alpha, 0.05, 0.5)
        return pd.Series(data).ewm(alpha=adaptive_alpha.mean()).mean()
    
    def atr(self, high, low, close, window=14):
        """Average True Range"""
        high = pd.Series(high)
        low = pd.Series(low)
        close = pd.Series(close)
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window).mean()
    
    def z_score(self, data, window=20):
        """Z-Score normalization"""
        series = pd.Series(data)
        mean = series.rolling(window).mean()
        std = series.rolling(window).std()
        return (series - mean) / std
    
    def bollinger_bands(self, data, window=20, num_std=2):
        """Bollinger Bands"""
        series = pd.Series(data)
        sma = series.rolling(window).mean()
        std = series.rolling(window).std()
        upper = sma + (std * num_std)
        lower = sma - (std * num_std)
        return upper, sma, lower
    
    def momentum_a(self, data, period=10):
        """Price Momentum"""
        return pd.Series(data).pct_change(period) * 100
    
    def momentum_v(self, data, volume, period=10):
        """Volume-adjusted Momentum"""
        price_change = pd.Series(data).pct_change(period)
        volume_avg = pd.Series(volume).rolling(period).mean()
        volume_ratio = pd.Series(volume) / volume_avg
        return price_change * volume_ratio * 100
    
    def fibonacci_retracement(self, high, low):
        """Fibonacci Retracement Levels"""
        diff = high - low
        return {
            '0%': high,
            '23.6%': high - 0.236 * diff,
            '38.2%': high - 0.382 * diff,
            '50%': high - 0.5 * diff,
            '61.8%': high - 0.618 * diff,
            '78.6%': high - 0.786 * diff,
            '100%': low
        }
    
    def fibonacci_extension(self, high, low):
        """Fibonacci Extension Levels"""
        diff = high - low
        return {
            '138.2%': high + 0.382 * diff,
            '161.8%': high + 0.618 * diff,
            '200%': high + 1.0 * diff
        }
    
    def steam_detection(self, prices, volume, threshold=2.0):
        """Steam Movement Detection"""
        price_change = pd.Series(prices).pct_change()
        volume_change = pd.Series(volume).pct_change()
        # Steam: large price change with increasing volume
        steam_signal = (abs(price_change) > threshold * price_change.std()) & (volume_change > 0)
        return steam_signal.astype(int)
    
    def rlm_detection(self, prices, volume, threshold=1.5):
        """Reverse Line Movement Detection"""
        price_change = pd.Series(prices).pct_change()
        volume_change = pd.Series(volume).pct_change()
        # RLM: price moves against volume trend
        rlm_signal = (price_change * volume_change < 0) & (abs(price_change) > threshold * price_change.std())
        return rlm_signal.astype(int)
    
    def flm_forecast(self, prices, volume, period=5):
        """Forward Line Movement Forecast"""
        # Use recent momentum and volume to forecast next movement
        recent_prices = prices[-period:]
        recent_volume = volume[-period:] if len(volume) >= period else volume
        
        if len(recent_prices) < 2:
            return 0
        
        price_trend = np.polyfit(range(len(recent_prices)), recent_prices, 1)[0]
        volume_trend = np.polyfit(range(len(recent_volume)), recent_volume, 1)[0] if len(recent_volume) > 1 else 0
        
        # Combine trends for forecast
        forecast = price_trend * (1 + 0.1 * volume_trend)
        return forecast
    
    def greek_analysis(self, prices, volatility_window=20):
        """Greek-like Sensitivity Analysis"""
        returns = np.diff(prices)
        volatility = pd.Series(returns).rolling(volatility_window).std().iloc[-1] if len(returns) > 0 else 0
        
        # Delta: sensitivity to price changes
        delta = np.polyfit(range(len(prices)), prices, 1)[0] if len(prices) > 1 else 0
        
        # Gamma: rate of change of delta (convexity)
        if len(prices) > 2:
            second_deriv = np.polyfit(range(len(prices)), prices, 2)[0] * 2
            gamma = second_deriv
        else:
            gamma = 0
        
        # Theta: time decay (not applicable directly, using recent momentum)
        theta = -abs(delta) * 0.01  # Small decay factor
        
        return {
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': volatility  # Sensitivity to volatility
        }

# ---------------------------
# ENHANCED ANALYSIS ENGINE
# ---------------------------

class EnhancedAnalysis:
    def __init__(self, window=20):
        self.ta = TAStack(window)
        self.window = window
    
    def analyze_dataset(self, data):
        """Comprehensive TA analysis on dataset"""
        if len(data) < 2:
            return data
        
        prices = [d['spread'] for d in data]
        # Use spread_vig as proxy for volume/market activity
        volume = [abs(d.get('spread_vig', 0)) for d in data]
        
        for i, point in enumerate(data):
            if i < self.window:
                # Initialize TA values for early data points
                self._initialize_ta_values(point)
                continue
            
            # Get data window for analysis
            window_data = data[max(0, i-self.window):i+1]
            window_prices = [d['spread'] for d in window_data]
            window_volume = [abs(d.get('spread_vig', 0)) for d in window_data]
            
            # Primary Drivers
            point['adaptive_ma'] = self.ta.adaptive_ma(window_prices).iloc[-1]
            point['ev'] = calculateEV(0.5, point.get('spread_vig', -110))  # Base EV
            
            # Secondary Indicators
            point['sma'] = self.ta.sma(window_prices).iloc[-1]
            point['ema'] = self.ta.ema(window_prices).iloc[-1]
            point['z_score'] = self.ta.z_score(window_prices).iloc[-1]
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = self.ta.bollinger_bands(window_prices)
            point['bb_upper'] = bb_upper.iloc[-1]
            point['bb_middle'] = bb_middle.iloc[-1]
            point['bb_lower'] = bb_lower.iloc[-1]
            point['bb_position'] = (window_prices[-1] - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1])
            
            # ATR
            high = max(window_prices)
            low = min(window_prices)
            point['atr'] = self.ta.atr([high], [low], window_prices).iloc[-1] if len(window_prices) > 1 else 0
            
            # Momentum
            point['momentum_a'] = self.ta.momentum_a(window_prices).iloc[-1] if len(window_prices) > 10 else 0
            point['momentum_v'] = self.ta.momentum_v(window_prices, window_volume).iloc[-1] if len(window_prices) > 10 else 0
            
            # Fibonacci
            fib_levels = self.ta.fibonacci_retracement(high, low)
            point['fib_236'] = fib_levels['23.6%']
            point['fib_382'] = fib_levels['38.2%']
            point['fib_618'] = fib_levels['61.8%']
            
            # Market Detection
            point['steam_signal'] = self.ta.steam_detection(window_prices, window_volume).iloc[-1]
            point['rlm_signal'] = self.ta.rlm_detection(window_prices, window_volume).iloc[-1]
            point['flm_forecast'] = self.ta.flm_forecast(window_prices, window_volume)
            
            # Greek Analysis
            greeks = self.ta.greek_analysis(window_prices)
            point.update(greeks)
            
            # Enhanced Kelly Criterion with TA adjustments
            base_prob = 0.5
            ta_adjustment = self._calculate_ta_adjustment(point)
            enhanced_prob = max(0.3, min(0.7, base_prob + ta_adjustment))
            point['enhanced_kelly'] = calculateKelly(enhanced_prob, point.get('spread_vig', -110))
            point['enhanced_ev'] = calculateEV(enhanced_prob, point.get('spread_vig', -110))
            
            # Combined Signal Score
            point['combined_signal'] = self._calculate_combined_signal(point)
        
        return data
    
    def _initialize_ta_values(self, point):
        """Initialize TA values for data points without enough history"""
        base_values = {
            'adaptive_ma': point['spread'], 'sma': point['spread'], 'ema': point['spread'],
            'z_score': 0, 'bb_upper': point['spread'], 'bb_middle': point['spread'],
            'bb_lower': point['spread'], 'bb_position': 0.5, 'atr': 0, 'momentum_a': 0,
            'momentum_v': 0, 'fib_236': point['spread'], 'fib_382': point['spread'],
            'fib_618': point['spread'], 'steam_signal': 0, 'rlm_signal': 0, 'flm_forecast': 0,
            'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'enhanced_kelly': 0,
            'enhanced_ev': 0, 'combined_signal': 0
        }
        point.update(base_values)
    
    def _calculate_ta_adjustment(self, point):
        """Calculate probability adjustment based on TA signals"""
        adjustment = 0
        
        # Trend following (adaptive MA)
        if point['spread'] > point['adaptive_ma']:
            adjustment += 0.05
        else:
            adjustment -= 0.05
        
        # Momentum
        adjustment += point['momentum_a'] * 0.01
        
        # Bollinger Band position
        if point['bb_position'] > 0.8:  # Near upper band
            adjustment -= 0.03
        elif point['bb_position'] < 0.2:  # Near lower band
            adjustment += 0.03
        
        # Z-score extreme
        if abs(point['z_score']) > 2:
            adjustment -= np.sign(point['z_score']) * 0.02
        
        return adjustment
    
    def _calculate_combined_signal(self, point):
        """Calculate combined signal strength from all TA indicators"""
        signals = []
        
        # Primary drivers (weight: 0.3 each)
        signals.append(np.sign(point['enhanced_ev']) * 0.3)
        signals.append(np.sign(point['flm_forecast']) * 0.3)
        
        # Confirmation indicators (weight: 0.1 each)
        signals.append(np.sign(point['momentum_a']) * 0.1)
        signals.append(np.sign(point['z_score']) * 0.1)
        signals.append((point['bb_position'] - 0.5) * 0.2)  # Band position
        
        # Market detection (weight: 0.15 each)
        signals.append(point['steam_signal'] * 0.15)
        signals.append(-point['rlm_signal'] * 0.15)  # RLM is bearish
        
        return sum(signals)

# ---------------------------
# BACKTESTING ENGINE
# ---------------------------

class BacktestingEngine:
    def __init__(self, initial_bankroll=10000):
        self.initial_bankroll = initial_bankroll
    
    def run_backtest(self, data, risk_multiplier=1.0):
        """Run comprehensive backtest with TA signals"""
        if len(data) < 10:
            return {'error': 'Insufficient data for backtesting'}
        
        bankroll = self.initial_bankroll
        bets_placed = 0
        wins = 0
        bet_history = []
        peak_bankroll = bankroll
        max_drawdown = 0
        
        for i, point in enumerate(data[:-1]):  # Don't bet on last point
            if point['combined_signal'] > 0.2 and point['enhanced_ev'] > 0:
                # Calculate bet size using Kelly with risk management
                kelly_fraction = point['enhanced_kelly'] * risk_multiplier
                bet_size = bankroll * min(kelly_fraction, 0.05)  # Max 5% of bankroll
                
                if bet_size < 10:  # Minimum bet size
                    continue
                
                # Simulate bet outcome with TA-adjusted probability
                base_win_prob = 0.5 + (point['combined_signal'] * 0.15)
                win_prob = max(0.4, min(0.8, base_win_prob))
                
                # Place bet
                if np.random.random() < win_prob:
                    bankroll += bet_size * 0.91  # -110 odds
                    outcome = 'Win'
                    wins += 1
                else:
                    bankroll -= bet_size
                    outcome = 'Loss'
                
                bets_placed += 1
                
                # Track drawdown
                if bankroll > peak_bankroll:
                    peak_bankroll = bankroll
                drawdown = (peak_bankroll - bankroll) / peak_bankroll
                max_drawdown = max(max_drawdown, drawdown)
                
                bet_history.append({
                    'timestamp': point['time'],
                    'team': point['team'],
                    'bet_size': bet_size,
                    'outcome': outcome,
                    'bankroll': bankroll,
                    'signal_strength': point['combined_signal'],
                    'ev': point['enhanced_ev']
                })
        
        # Calculate performance metrics
        profit = bankroll - self.initial_bankroll
        roi = (profit / self.initial_bankroll) * 100
        win_rate = (wins / bets_placed * 100) if bets_placed > 0 else 0
        
        # Sharpe ratio (simplified)
        if len(bet_history) > 1:
            returns = [b['bankroll'] / self.initial_bankroll - 1 for b in bet_history]
            sharpe = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
        else:
            sharpe = 0
        
        return {
            'initial_bankroll': self.initial_bankroll,
            'final_bankroll': bankroll,
            'profit': profit,
            'roi': roi,
            'bets_placed': bets_placed,
            'win_rate': win_rate,
            'max_drawdown': max_drawdown * 100,
            'sharpe_ratio': sharpe,
            'bet_history': bet_history
        }

# ---------------------------
# DATA PROCESSING
# ---------------------------

def parse_blocks(raw_text):
    """Parse the 5-line block format"""
    lines = [l.strip() for l in raw_text.split('\n') if l.strip()]
    rows = []
    
    for i in range(0, len(lines), 5):
        if i + 4 >= len(lines):
            break
            
        try:
            L1, L2, L3, L4, L5 = lines[i:i+5]
            tks = L1.split()
            
            if len(tks) < 4:
                continue
                
            # Parse date/time
            date_part = tks[0]
            time_part = tks[1]
            team = ' '.join(tks[2:-1])
            spread_raw = tks[-1]
            
            # Parse datetime
            month, day = map(int, date_part.split('/'))
            time_clean = time_part[:-2]
            hour, minute = map(int, time_clean.split(':'))
            period = time_part[-2:].upper()
            
            if period == 'PM' and hour < 12:
                hour += 12
            if period == 'AM' and hour == 12:
                hour = 0
                
            dt = datetime(2025, month, day, hour, minute)
            
            # Parse values
            spread = float(spread_raw) if spread_raw.lower() != 'even' else 0
            spread_vig = float(L2) if L2.lower() != 'even' else 100
            total_side = L3[0].lower() if L3 else 'o'
            total = float(L3[1:]) if L3 and L3[1:] else 0
            total_vig = float(L4) if L4.lower() != 'even' else 100
            
            ml_tokens = L5.split()
            ml_away = float(ml_tokens[0]) if ml_tokens and ml_tokens[0].lower() != 'even' else 100
            ml_home = float(ml_tokens[1]) if len(ml_tokens) > 1 and ml_tokens[1].lower() != 'even' else 100
            
            rows.append({
                'time': dt,
                'team': team,
                'spread': spread,
                'spread_vig': spread_vig,
                'total': total,
                'total_side': total_side,
                'total_vig': total_vig,
                'ml_away': ml_away,
                'ml_home': ml_home
            })
            
        except Exception as e:
            continue
    
    if rows:
        rows.sort(key=lambda x: x['time'])
    return rows

# ---------------------------
# STREAMLIT APP
# ---------------------------

def main():
    st.set_page_config(
        page_title="Omniscience - Complete TA Stack",
        layout="wide",
        page_icon="🧠",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header { font-size: 2.5rem; color: #1f77b4; text-align: center; margin-bottom: 2rem; }
    .metric-card { background: #f0f2f6; padding: 1rem; border-radius: 10px; margin: 0.5rem 0; }
    .positive { color: #00d154; font-weight: bold; }
    .negative { color: #ff4b4b; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="main-header">🧠 Omniscience - Complete TA Stack</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        bankroll = st.number_input("Bankroll ($)", value=10000.0, min_value=100.0, step=500.0)
        risk_multiplier = st.slider("Risk Multiplier", 0.1, 3.0, 1.0, 0.1)
        ta_window = st.slider("TA Analysis Window", 5, 50, 20)
        
        st.markdown("---")
        st.subheader("📋 Data Format")
        st.code("""
10/25 10:30AM Lakers -3.5
-110
o44.5
-110
-150 130

10/25 11:00AM Lakers -4.0
-115
o45.0
-105
-160 140
        """)
        
        st.markdown("---")
        st.subheader("🔧 TA Stack Status")
        st.success("✅ Full TA Stack Active")
        st.info("""
        **Primary Drivers:**
        - Adaptive MA, EV Analysis
        
        **Confirmation Indicators:**
        - SMA, EMA, Bollinger Bands, Z-Score, ATR
        
        **Momentum & Forecasting:**
        - Momentum A/V, Fibonacci, FLM Forecast
        
        **Market Detection:**
        - Steam, RLM Detection
        
        **Risk Management:**
        - Greek Analysis, Kelly Criterion
        """)
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        raw_data = st.text_area(
            "📥 Paste Odds Data (5-line blocks)",
            height=300,
            placeholder="Paste your odds data here in 5-line blocks format...",
            key="data_input"
        )
    
    with col2:
        st.subheader("🚀 Analysis Controls")
        
        if st.button("🧠 Run Comprehensive TA Analysis", type="primary", use_container_width=True):
            if raw_data.strip():
                with st.spinner("Running full TA stack analysis..."):
                    # Parse data
                    parsed_data = parse_blocks(raw_data)
                    
                    if not parsed_data:
                        st.error("❌ No valid data blocks found")
                    else:
                        # Run TA analysis
                        analyzer = EnhancedAnalysis(ta_window)
                        analyzed_data = analyzer.analyze_dataset(parsed_data)
                        
                        # Generate recommendations
                        recommendations = generate_recommendations(analyzed_data, bankroll)
                        
                        st.success("✅ Analysis Complete!")
                        
                        # Display results
                        display_analysis_results(analyzed_data, recommendations)
            else:
                st.warning("⚠️ Please enter some data to analyze")
        
        if st.button("📊 Run Advanced Backtest", use_container_width=True):
            if raw_data.strip():
                with st.spinner("Running advanced backtest..."):
                    parsed_data = parse_blocks(raw_data)
                    if parsed_data:
                        analyzer = EnhancedAnalysis(ta_window)
                        analyzed_data = analyzer.analyze_dataset(parsed_data)
                        
                        backtester = BacktestingEngine(bankroll)
                        results = backtester.run_backtest(analyzed_data, risk_multiplier)
                        
                        display_backtest_results(results)
                    else:
                        st.error("❌ No valid data for backtesting")
            else:
                st.warning("⚠️ Please enter some data to backtest")
    
    # Detailed analysis section
    if raw_data.strip():
        with st.expander("🔍 Detailed TA Metrics", expanded=False):
            parsed_data = parse_blocks(raw_data)
            if parsed_data:
                analyzer = EnhancedAnalysis(ta_window)
                analyzed_data = analyzer.analyze_dataset(parsed_data)
                display_detailed_metrics(analyzed_data)

def generate_recommendations(data, bankroll):
    """Generate trading recommendations based on TA signals"""
    if not data:
        return {'recommendation': 'HOLD', 'confidence': 0, 'details': {}}
    
    strong_bets = 0
    bets = 0
    avoids = 0
    strong_avoids = 0
    
    for point in data:
        signal = point.get('combined_signal', 0)
        if signal > 0.5:
            strong_bets += 1
        elif signal > 0.1:
            bets += 1
        elif signal < -0.5:
            strong_avoids += 1
        elif signal < -0.1:
            avoids += 1
    
    total_signals = strong_bets + bets + avoids + strong_avoids
    net_score = (strong_bets * 2 + bets) - (strong_avoids * 2 + avoids)
    
    if net_score > 3:
        recommendation = 'STRONG BUY'
        confidence = min(90, (net_score / total_signals) * 100) if total_signals > 0 else 0
    elif net_score > 0:
        recommendation = 'BUY'
        confidence = min(70, (net_score / total_signals) * 100) if total_signals > 0 else 0
    elif net_score < -3:
        recommendation = 'STRONG SELL'
        confidence = min(90, (abs(net_score) / total_signals) * 100) if total_signals > 0 else 0
    elif net_score < 0:
        recommendation = 'SELL'
        confidence = min(70, (abs(net_score) / total_signals) * 100) if total_signals > 0 else 0
    else:
        recommendation = 'HOLD'
        confidence = 0
    
    return {
        'recommendation': recommendation,
        'confidence': confidence,
        'details': {
            'strong_bets': strong_bets,
            'bets': bets,
            'strong_avoids': strong_avoids,
            'avoids': avoids,
            'net_score': net_score
        }
    }

def display_analysis_results(data, recommendations):
    """Display analysis results in a comprehensive format"""
    st.subheader("📈 TA Analysis Results")
    
    # Recommendation card
    rec_color = "#00d154" if "BUY" in recommendations['recommendation'] else "#ff4b4b" if "SELL" in recommendations['recommendation'] else "#64748b"
    st.markdown(f"""
    <div style="background: {rec_color}20; padding: 1rem; border-radius: 10px; border-left: 4px solid {rec_color};">
        <h3 style="color: {rec_color}; margin: 0;">🎯 {recommendations['recommendation']}</h3>
        <p style="margin: 0.5rem 0 0 0;">Confidence: <b>{recommendations['confidence']:.1f}%</b></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        latest = data[-1] if data else {}
        ev_color = "positive" if latest.get('enhanced_ev', 0) > 0 else "negative"
        st.metric("Enhanced EV", f"{latest.get('enhanced_ev', 0):.3f}", delta=None)
    
    with col2:
        st.metric("Kelly Fraction", f"{latest.get('enhanced_kelly', 0):.3f}")
    
    with col3:
        st.metric("Signal Strength", f"{latest.get('combined_signal', 0):.3f}")
    
    with col4:
        st.metric("Momentum A", f"{latest.get('momentum_a', 0):.2f}%")
    
    # Price chart with TA indicators
    if len(data) > 1:
        fig = create_ta_chart(data)
        st.plotly_chart(fig, use_container_width=True)

def display_backtest_results(results):
    """Display backtest results"""
    st.subheader("📊 Backtest Results")
    
    if 'error' in results:
        st.error(results['error'])
        return
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        profit_color = "positive" if results['profit'] >= 0 else "negative"
        st.metric("Profit/Loss", f"${results['profit']:,.2f}", delta=None)
    
    with col2:
        roi_color = "positive" if results['roi'] >= 0 else "negative"
        st.metric("ROI", f"{results['roi']:.2f}%")
    
    with col3:
        st.metric("Win Rate", f"{results['win_rate']:.1f}%")
    
    with col4:
        st.metric("Max Drawdown", f"{results['max_drawdown']:.1f}%")
    
    # Additional metrics
    col5, col6, col7, col8 = st.columns(4)
    
    with col5:
        st.metric("Bets Placed", results['bets_placed'])
    
    with col6:
        st.metric("Sharpe Ratio", f"{results['sharpe_ratio']:.2f}")
    
    with col7:
        st.metric("Final Bankroll", f"${results['final_bankroll']:,.2f}")
    
    with col8:
        st.metric("Risk Multiplier", "Custom")

def display_detailed_metrics(data):
    """Display detailed TA metrics in a dataframe"""
    if not data:
        return
    
    df = pd.DataFrame(data)
    
    # Format datetime for display
    if 'time' in df.columns:
        df['time'] = df['time'].dt.strftime('%m/%d %H:%M')
    
    # Select key metrics to display
    ta_columns = [col for col in df.columns if col not in ['team', 'total_side', 'ml_away', 'ml_home']]
    
    st.dataframe(
        df[ta_columns].style.format({
            'spread': '{:.1f}',
            'spread_vig': '{:.0f}',
            'total': '{:.1f}',
            'total_vig': '{:.0f}',
            'adaptive_ma': '{:.2f}',
            'sma': '{:.2f}',
            'ema': '{:.2f}',
            'z_score': '{:.2f}',
            'atr': '{:.3f}',
            'momentum_a': '{:.2f}',
            'momentum_v': '{:.2f}',
            'enhanced_ev': '{:.3f}',
            'enhanced_kelly': '{:.3f}',
            'combined_signal': '{:.3f}',
            'delta': '{:.3f}',
            'gamma': '{:.3f}'
        }),
        use_container_width=True,
        height=400
    )

def create_ta_chart(data):
    """Create interactive chart with TA indicators"""
    if len(data) < 2:
        return go.Figure()
    
    times = [d['time'] for d in data]
    spreads = [d['spread'] for d in data]
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Price with TA Indicators', 'Momentum & Signals'),
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3]
    )
    
    # Price and moving averages
    fig.add_trace(go.Scatter(x=times, y=spreads, name='Spread', line=dict(color='#1f77b4')), row=1, col=1)
    
    if 'adaptive_ma' in data[0]:
        adaptive_ma = [d['adaptive_ma'] for d in data]
        fig.add_trace(go.Scatter(x=times, y=adaptive_ma, name='Adaptive MA', line=dict(color='#ff7f0e')), row=1, col=1)
    
    if 'bb_upper' in data[0]:
        bb_upper = [d['bb_upper'] for d in data]
        bb_lower = [d['bb_lower'] for d in data]
        fig.add_trace(go.Scatter(x=times, y=bb_upper, name='BB Upper', line=dict(color='rgba(255,0,0,0.3)'), showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=times, y=bb_lower, name='BB Lower', line=dict(color='rgba(0,255,0,0.3)'), showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=times+times[::-1], y=bb_upper+bb_lower[::-1], fill='toself', name='Bollinger Band', fillcolor='rgba(0,100,80,0.2)', line=dict(color='rgba(255,255,255,0)')), row=1, col=1)
    
    # Momentum and signals
    if 'momentum_a' in data[0]:
        momentum = [d['momentum_a'] for d in data]
        fig.add_trace(go.Scatter(x=times, y=momentum, name='Momentum A', line=dict(color='#2ca02c')), row=2, col=1)
    
    if 'combined_signal' in data[0]:
        signals = [d['combined_signal'] for d in data]
        fig.add_trace(go.Scatter(x=times, y=signals, name='Combined Signal', line=dict(color='#d62728')), row=2, col=1)
    
    fig.update_layout(height=600, showlegend=True, template="plotly_white")
    return fig

if __name__ == "__main__":
    main()
