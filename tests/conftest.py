# tests/conftest.py
import pytest
import xarray as xr
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString

@pytest.fixture
def fake_ensemble():
    """
    Creates a small synthetic dataset for testing.
    """
    # Create dimensions
    members = [1, 2, 3] 
    times = pd.date_range("2023-01-01", periods=5, freq="D")
    
    data = np.zeros((len(members), len(times)))
    for i, m in enumerate(members):
        data[i, :] = float(m)
        
    ds = xr.Dataset(
        data_vars={
            "streamflow": (("Formulation_ID", "time"), data),
            "velocity": (("Formulation_ID", "time"), data * 10),
        },
        coords={
            "Formulation_ID": members,
            "time": times,
        }
    )
    return ds

@pytest.fixture
def fake_spatial_ensemble():
    """
    Spatial synthetic dataset for VIZ tests.
    """
    members = [1, 2, 3] 
    times = pd.date_range("2023-01-01", periods=5, freq="D")
    feature_ids = [12345, 67890] 
    shape = (len(members), len(times), len(feature_ids))
    data = np.zeros(shape)
    
    for i, m in enumerate(members):
        data[i, :, :] = float(m)
        
    ds = xr.Dataset(
        data_vars={
            "streamflow": (("Formulation_ID", "time", "feature_id"), data),
            "velocity": (("Formulation_ID", "time", "feature_id"), data * 10),
        },
        coords={
            "Formulation_ID": members,
            "time": times,
            "feature_id": feature_ids
        }
    )
    return ds

@pytest.fixture
def fake_hydrofabric():
    """
    Creates a simple GeoDataFrame matching the fake ensemble IDs.
    """
    # Create 2 simple lines
    data = {
        'feature_id': [12345, 67890],
        'geometry': [
            LineString([(0, 0), (1, 1)]),
            LineString([(1, 1), (2, 2)])
        ]
    }
    gdf = gpd.GeoDataFrame(data, crs="EPSG:4326")
    return gdf