import pytest
import os
from pathlib import Path
import xarray as xr
import geopandas as gpd
from teval import io

# --- Helper: Locate Sample Data ---
# This finds the 'data' folder relative to this test file.
# Assumes structure:
#   project_root/
#     data/
#     tests/test_io.py
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"

# We check if data exists; if not, we skip these tests automatically
# to prevent CI/CD failures on machines without your local data.
has_gpkg = list(DATA_DIR.glob("*.gpkg"))
has_nc = list(DATA_DIR.glob("*.nc"))

@pytest.mark.skipif(not has_gpkg, reason="No .gpkg files found in data/")
def test_load_hydrofabric_real_file():
    """
    Test loading the real hydrofabric geopackage.
    """
    # Grab the first gpkg found in data/
    gpkg_path = list(DATA_DIR.glob("*.gpkg"))[0]
    
    print(f"Testing load with: {gpkg_path}")
    gdf = io.load_hydrofabric(gpkg_path)
    
    assert isinstance(gdf, gpd.GeoDataFrame)
    assert not gdf.empty
    # Check for expected columns often found in NextGen hydrofabric
    assert "feature_id" in gdf.columns or "id" in gdf.columns

@pytest.mark.skipif(not has_nc, reason="No .nc files found in data/")
def test_load_ensemble_real_files():
    """
    Test loading the real t-route output NetCDF files.
    """
    # Construct the glob pattern (e.g., data/ensemble_member_*.nc)
    pattern = str(DATA_DIR / "*.nc")
    
    print(f"Testing load with pattern: {pattern}")
    ds = io.load_ensemble(pattern)
    
    assert isinstance(ds, xr.Dataset)
    # Check standard dimensions
    assert "ensemble_member" in ds.dims
    assert "time" in ds.dims
    assert "feature_id" in ds.dims
    
    # Ensure we actually loaded variables
    assert "streamflow" in ds.data_vars

def test_save_ensemble_stats(fake_ensemble, tmp_path):
    """
    Test saving a file. We use 'fake_ensemble' (from conftest.py) 
    so this test runs even if you don't have real data.
    
    'tmp_path' is a pytest fixture that provides a temporary directory 
    unique to this test invocation.
    """
    # Create a dummy output path in the temp dir
    output_file = tmp_path / "test_output.nc"
    
    # Run the save function
    io.save_ensemble_stats(fake_ensemble, output_file)
    
    # Verify file exists
    assert output_file.exists()
    
    # Verify we can read it back
    ds_loaded = xr.open_dataset(output_file)
    assert "streamflow" in ds_loaded.data_vars
    assert ds_loaded.attrs == fake_ensemble.attrs
    ds_loaded.close()