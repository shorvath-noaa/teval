import pytest
import xarray as xr
import numpy as np
import pandas as pd
from teval.ensemble import SimpleEnsembler

# --- Fixtures ---
@pytest.fixture
def dummy_ensemble():
    """Creates a small dummy ensemble dataset for testing."""
    times = pd.date_range("2023-01-01", periods=5, freq="D")
    members = [1, 2, 3]
    feature_ids = [101, 102]
    
    # Create random data: (time, member, feature)
    # We make it deterministic for testing math
    data = np.zeros((5, 3, 2))
    data[:, 0, :] = 10  # Member 1 is always 10
    data[:, 1, :] = 20  # Member 2 is always 20
    data[:, 2, :] = 30  # Member 3 is always 30
    
    ds = xr.Dataset(
        data_vars={
            "streamflow": (("time", "member", "feature_id"), data)
        },
        coords={
            "time": times,
            "member": members,
            "feature_id": feature_ids
        }
    )
    return ds

# --- Tests ---

def test_simple_ensembler_mean_median(dummy_ensemble):
    """Test that SimpleEnsembler correctly calculates mean and median."""
    # Setup
    ensembler = SimpleEnsembler(quantiles=[])
    
    # Act
    ds_out = ensembler.process(dummy_ensemble)
    
    # Assert
    assert "streamflow_mean" in ds_out.data_vars
    assert "streamflow_median" in ds_out.data_vars
    
    # Check values
    # Mean of [10, 20, 30] is 20
    assert np.all(ds_out["streamflow_mean"] == 20)
    # Median of [10, 20, 30] is 20
    assert np.all(ds_out["streamflow_median"] == 20)
    
    # Ensure member dimension is gone
    assert "member" not in ds_out.dims

def test_simple_ensembler_quantiles(dummy_ensemble):
    """Test that quantiles are generated and named correctly."""
    # Setup: Ask for min (0.0) and max (1.0) to make checking easy
    ensembler = SimpleEnsembler(quantiles=[0.0, 1.0])
    
    # Act
    ds_out = ensembler.process(dummy_ensemble)
    
    # Assert
    assert "streamflow_p00" in ds_out.data_vars
    assert "streamflow_p100" in ds_out.data_vars
    
    # Min of [10, 20, 30] is 10
    assert np.all(ds_out["streamflow_p00"] == 10)
    # Max of [10, 20, 30] is 30
    assert np.all(ds_out["streamflow_p100"] == 30)

def test_missing_dimension_error():
    """Test that it raises an error if the member dimension is missing."""
    ds_bad = xr.Dataset({"data": (("time", "x"), np.zeros((2, 2)))})
    ensembler = SimpleEnsembler()
    
    with pytest.raises(ValueError, match="Could not identify ensemble member dimension"):
        ensembler.process(ds_bad)