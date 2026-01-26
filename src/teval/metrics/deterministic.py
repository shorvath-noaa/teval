# src/teval/metrics/deterministic.py
import numpy as np
import pandas as pd
import xarray as xr
from typing import Union, Dict

def align_and_validate(sim: Union[pd.Series, xr.DataArray], obs: pd.Series) -> pd.DataFrame:
    """
    Aligns simulation and observation data by time index.
    Removes NaNs from both series to ensure fair comparison.
    """
    # Convert xarray to pandas if needed
    if isinstance(sim, xr.DataArray):
        sim = sim.to_series()

    # Align indices (inner join on time)
    # This handles mismatched start/end dates automatically
    df = pd.DataFrame({'sim': sim, 'obs': obs}).dropna()

    if df.empty:
        raise ValueError("No overlapping valid data between simulation and observation.")
    
    return df

def nse(sim: Union[pd.Series, xr.DataArray], obs: pd.Series) -> float:
    """
    Nash-Sutcliffe Efficiency (NSE).
    Range: (-inf, 1]. 1 is perfect. 0 is as good as the mean of obs.
    """
    df = align_and_validate(sim, obs)
    s = df['sim'].values
    o = df['obs'].values
    
    numerator = np.sum((s - o) ** 2)
    denominator = np.sum((o - np.mean(o)) ** 2)
    
    if denominator == 0:
        return -np.inf # Variance of obs is zero
        
    return 1 - (numerator / denominator)

def kge(sim: Union[pd.Series, xr.DataArray], obs: pd.Series) -> float:
    """
    Kling-Gupta Efficiency (KGE).
    Range: (-inf, 1]. 1 is perfect.
    """
    df = align_and_validate(sim, obs)
    s = df['sim'].values
    o = df['obs'].values

    # Mean and Std Dev
    mean_s, mean_o = np.mean(s), np.mean(o)
    std_s, std_o = np.std(s), np.std(o)
    
    # Components
    r = np.corrcoef(s, o)[0, 1] # Correlation
    alpha = std_s / std_o       # Variability ratio
    beta = mean_s / mean_o      # Bias ratio
    
    # KGE formula
    kge_val = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
    
    return kge_val

def rmse(sim: Union[pd.Series, xr.DataArray], obs: pd.Series) -> float:
    """
    Root Mean Square Error.
    Lower is better.
    """
    df = align_and_validate(sim, obs)
    s = df['sim'].values
    o = df['obs'].values
    
    return np.sqrt(np.mean((s - o)**2))

def calculate_all(sim: Union[pd.Series, xr.DataArray], obs: pd.Series) -> Dict[str, float]:
    """Convenience function to return a dict of all metrics."""
    return {
        "NSE": nse(sim, obs),
        "KGE": kge(sim, obs),
        "RMSE": rmse(sim, obs)
    }