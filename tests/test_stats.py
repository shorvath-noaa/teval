import pytest
import xarray as xr
import numpy as np
import pandas as pd
from teval import stats

def test_calculate_basics_mean(fake_ensemble):
    ds = fake_ensemble
    stats_ds = stats.calculate_basics(ds)
    
    # Mean of [1, 2, 3] is 2.0
    expected_mean = 2.0
    actual_mean = stats_ds["streamflow_mean"].isel(time=0).values
    
    assert actual_mean == expected_mean, f"Expected mean 2.0, got {actual_mean}"

def test_calculate_basics_structure(fake_ensemble):
    ds = fake_ensemble
    stats_ds = stats.calculate_basics(ds)
    
    for suffix in ["_mean", "_median", "_std", "_p05", "_p95"]:
        assert f"streamflow{suffix}" in stats_ds