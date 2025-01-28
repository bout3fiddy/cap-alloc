"""
Utility functions for time series aggregation.
"""

import pandas as pd
import numpy as np

def resample_to_epochs(data, epoch_length: float):
    """Resample data to specified epoch length in days.
    
    @param data: Input time series data
    @param epoch_length: Length of each epoch in days (can be fractional)
    @return: Resampled dataframe
    """
    # Convert epoch_length (in days) to a pandas-compatible frequency string
    freq = f'{int(epoch_length)}D'
    
    return data.resample(freq, label='right').agg({
        'Net_APY': 'mean',
        'Supply_APY': 'mean',
        'Borrow_APY': 'mean',
        'Staking_APY': 'mean'
    })
    

def get_group(grouped_data: pd.core.resample.Resampler, group_id: int) -> pd.DataFrame:

    if not isinstance(grouped_data, pd.core.resample.Resampler):
        raise TypeError("grouped_data must be a pandas Resampler object")
    
    groups_list = list(grouped_data.groups.keys())
    n_groups = len(groups_list)
    
    if group_id >= n_groups:
        raise IndexError(f"Group ID {group_id} out of range. Max: {n_groups-1}")
    
    start_time = groups_list[group_id]
    end_time = groups_list[group_id + 1] if group_id + 1 < n_groups else None
    
    return grouped_data.get_group(start_time)

def continuous_ema(
    new_value: float,
    last_ema: float,
    time_delta: float,  # time passed between last update and current time (hours)
    averaging_window: float  # (hours)
) -> float:

    # Calculate continuous exponential moving average.
    
    if time_delta <= 0:
        return last_ema
    
    # Calculate decay factor using exponential decay formula
    alpha = np.exp(-time_delta / averaging_window)
    
    # Update EMA
    return new_value * (1 - alpha) + last_ema * alpha
