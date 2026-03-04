import numpy as np
from typing import Callable, Dict

def bootstrap_confidence_interval(
    obs: np.ndarray, 
    sim: np.ndarray, 
    metric_func: Callable, 
    n_samples: int = 1000, 
    ci_level: float = 0.95,
    min_samples: int = 5
) -> Dict[str, float]:
    """
    Performs bootstrapping to estimate Metric Confidence Interval (CI).
    Used for the 'Hypothesis Map' (Skillful/Unskillful/Indeterminate).
    """
    # 1. Align Data
    valid = np.isfinite(obs) & np.isfinite(sim)
    o_clean, s_clean = obs[valid], sim[valid]
    n = len(o_clean)
    
    if n < min_samples:
        return {'significance': 'insufficient_data'}

    # 2. Resample indices (Block bootstrapping preferred for time series, simple used here for speed)
    scores = []
    indices = np.random.randint(0, n, (n_samples, n))
    
    for idx_row in indices:
        scores.append(metric_func(o_clean[idx_row], s_clean[idx_row]))
    
    scores = np.array(scores)
    
    # 3. Calculate CI
    alpha = (1 - ci_level) / 2
    lower = np.nanpercentile(scores, alpha * 100)
    upper = np.nanpercentile(scores, (1 - alpha) * 100)
    median_score = np.nanpercentile(scores, 50)
    
    # 4. Classify Hypothesis
    # H0: Skill = 0. 
    if lower > 0:
        sig = "skillful"       # CI > 0
    elif upper < 0:
        sig = "unskillful"     # CI < 0
    else:
        sig = "indeterminate"  # CI spans 0

    return {
        'lower': lower,
        'upper': upper,
        'median': median_score,
        'significance': sig
    }