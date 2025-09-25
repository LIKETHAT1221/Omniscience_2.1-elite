import pandas as pd
import numpy as np

def track_line_movement(parsed_data, field, smooth_window=5):
    """
    Compute delta, momentum, MOM_A and MOM_V for a given field ('spread' or 'total').
    Adds:
        - {field}_delta
        - {field}_momentum
        - {field}_momentum_smooth
        - {field}_mom_a
        - {field}_mom_v
    """
    df = pd.DataFrame(parsed_data)
    if field not in df.columns:
        raise ValueError(f"{field} not in data")

    # Delta: difference from previous row
    df[f'{field}_delta'] = df[field].diff().fillna(0)
    # Momentum: delta of delta
    df[f'{field}_momentum'] = df[f'{field}_delta'].diff().fillna(0)
    # Smoothed momentum
    df[f'{field}_momentum_smooth'] = df[f'{field}_momentum'].rolling(window=smooth_window, min_periods=1).mean()
    # MOM_A: rolling mean of momentum (forecasting trend)
    df[f'{field}_mom_a'] = df[f'{field}_momentum'].rolling(window=smooth_window, min_periods=1).mean()
    # MOM_V: rolling std of momentum (forecasting stability)
    df[f'{field}_mom_v'] = df[f'{field}_momentum'].rolling(window=smooth_window, min_periods=1).std().fillna(0)

    # Return updated list of dicts
    return df.to_dict(orient='records')
