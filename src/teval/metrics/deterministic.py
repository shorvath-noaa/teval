import numpy as np
import pandas as pd

def _validate(obs, sim):
    """Aligns and filters NaN values."""
    valid = np.isfinite(obs) & np.isfinite(sim)
    return obs[valid], sim[valid]

# Standard Scores
def _nse(obs: np.ndarray, sim: np.ndarray) -> float:
    """Nash-Sutcliffe Efficiency."""
    o, s = _validate(obs, sim)
    if len(o) < 2: return np.nan
    denom = np.sum((o - np.mean(o))**2)
    if denom == 0: return np.nan
    return 1 - (np.sum((o - s)**2) / denom)

def _kge(obs: np.ndarray, sim: np.ndarray) -> float:
    """Kling-Gupta Efficiency."""
    o, s = _validate(obs, sim)
    if len(o) < 2: return np.nan
    
    mean_o, mean_s = np.mean(o), np.mean(s)
    std_o, std_s = np.std(o), np.std(s)
    if std_o < 1e-5 or mean_o < 1e-5 or std_s < 1e-5: 
        return np.nan
    
    r = np.corrcoef(o, s)[0, 1]
    alpha = std_s / std_o
    beta = mean_s / mean_o
    return 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)

def _pbias(obs: np.ndarray, sim: np.ndarray) -> float:
    """Percent Bias."""
    o, s = _validate(obs, sim)
    denom = np.sum(o)
    if denom == 0: 
        return np.nan
    return 100 * np.sum(s - o) / denom if len(o) > 0 else np.nan

# Hydrological Signatures
def _peak_flow_error(obs: np.ndarray, sim: np.ndarray) -> float:
    """Percent error in Peak Flow magnitude."""
    o_max = np.nanmax(obs)
    s_max = np.nanmax(sim)
    if o_max == 0: return np.nan
    return 100 * (s_max - o_max) / o_max

def _peak_timing_error(obs: pd.Series, sim: pd.Series) -> float:
    """
    Difference in hours between observed and simulated peak.
    """
    t_o = obs.idxmax()
    t_s = sim.idxmax()
    
    # Calculate difference in hours
    diff = (t_s - t_o).total_seconds() / 3600.0
    return diff

def calculate_deterministic_metric(obs: pd.Series, sim: pd.Series, metric_name: str) -> float:
    """Wrapper to calculate a specified deterministic metric."""
    metric_funcs = {
        'nse': _nse,
        'kge': _kge,
        'pbias': _pbias,
        'peak_flow_error': _peak_flow_error,
        'peak_timing_error': _peak_timing_error
    }
    
    func = metric_funcs.get(metric_name.lower())
    if func is None:
        raise ValueError(f"Unsupported metric: {metric_name}")
    
    # peak_timing_error explicitly requires Pandas Series (for the time index)
    if metric_name.lower() == 'peak_timing_error':
        return func(obs, sim)
        
    obs_arr = obs.values if hasattr(obs, 'values') else obs
    sim_arr = sim.values if hasattr(sim, 'values') else sim
    
    return func(obs_arr, sim_arr)