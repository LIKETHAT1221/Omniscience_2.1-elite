import pandas as pd
import numpy as np

def track_line_movement(parsed_data, field, smooth_window=5):
    """
    Calculate delta, momentum, smoothed momentum, MOM_A, and MOM_V
    for a given field (spread or total) across the entire parsed_data.
    """
    values = [row[field] for row in parsed_data]

    # Convert to Series for rolling calculations
    s = pd.Series(values, dtype=float)

    # Delta (difference from previous value)
    delta = s.diff().fillna(0)

    # Momentum (current delta)
    momentum = delta.copy()

    # Smoothed momentum (rolling mean)
    momentum_smooth = momentum.rolling(window=smooth_window, min_periods=1).mean()

    # MOM_A = rolling average of momentum (like adaptive MA for momentum)
    mom_a = momentum.rolling(window=smooth_window, min_periods=1).mean()

    # MOM_V = rolling volatility of momentum
    mom_v = momentum.rolling(window=smooth_window, min_periods=1).std().fillna(0)

    # Assign back to parsed_data
    for i, row in enumerate(parsed_data):
        row[f"{field}_delta"] = float(delta.iloc[i])
        row[f"{field}_momentum"] = float(momentum.iloc[i])
        row[f"{field}_momentum_smooth"] = float(momentum_smooth.iloc[i])
        row[f"{field}_mom_a"] = float(mom_a.iloc[i])
        row[f"{field}_mom_v"] = float(mom_v.iloc[i])

    return parsed_data
