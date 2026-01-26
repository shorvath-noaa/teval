# tests/conftest.py
import pytest
import xarray as xr
import numpy as np

@pytest.fixture
def fake_ensemble():
    """
    Creates a dummy 3-member ensemble for testing.
    Pytest will automatically pass the return value of this function 
    to any test function that asks for 'fake_ensemble'.
    """
    # 3 Members, 5 Time steps
    data = np.array([
        [1.0, 1.0, 1.0, 1.0, 1.0],  # Member 0
        [2.0, 2.0, 2.0, 2.0, 2.0],  # Member 1
        [3.0, 3.0, 3.0, 3.0, 3.0],  # Member 2
    ])
    
    ds = xr.Dataset(
        data_vars={
            "streamflow": (("ensemble_member", "time"), data),
            "velocity": (("ensemble_member", "time"), data * 10)
        },
        coords={
            "ensemble_member": [0, 1, 2],
            "time": [0, 1, 2, 3, 4]
        }
    )
    return ds