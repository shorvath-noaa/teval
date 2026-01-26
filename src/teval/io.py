import xarray as xr
import geopandas as gpd
import pandas as pd
import glob
from pathlib import Path
from typing import List, Union

def load_ensemble(
    file_pattern: str, 
    concat_dim: str = "ensemble_member"
) -> xr.Dataset:
    """
    Loads ensemble NetCDF files serially (no Dask parallelization) to avoid 
    HDF5 segmentation faults.
    
    Args:
        file_pattern: A glob pattern (e.g., "data/ensemble_member_*.nc").
        concat_dim: The name of the new dimension to stack files along.
        
    Returns:
        xr.Dataset: A combined dataset of all ensemble members.
    """
    # 1. Find all matching files
    files = sorted(glob.glob(file_pattern))
    
    if not files:
        raise FileNotFoundError(f"No files found matching pattern: {file_pattern}")
        
    print(f"Found {len(files)} ensemble files. Loading serially...")
    
    datasets = []
    
    # 2. Loop through and open each file
    for i, f in enumerate(files):
        try:
            # open_dataset is lazy by default (reads header, not data)
            ds = xr.open_dataset(f)
            
            # If the files don't already have an 'ensemble_member' coordinate,
            # we can add it on the fly so they stack correctly.
            if concat_dim not in ds.coords:
                ds = ds.expand_dims({concat_dim: [i]})
                
            datasets.append(ds)
            
        except Exception as e:
            print(f"Warning: Failed to load {f}. Error: {e}")

    if not datasets:
        raise RuntimeError("Could not load any valid NetCDF files.")

    # 3. Concatenate into one object
    # This keeps the underlying file handles open but doesn't read values yet
    combined_ds = xr.concat(datasets, dim=concat_dim)
    
    # Standardize coordinate names if necessary
    if 'feature_id' in combined_ds.dims:
        # Ensure feature_id is the index
        if combined_ds.indexes.get('feature_id') is None:
             combined_ds = combined_ds.set_index(feature_id="feature_id")

    return combined_ds

def load_hydrofabric(
    gpkg_path: str, 
    layer: str = "flowpaths", 
    id_col: str = "id",
    match_netcdf_ids: bool = True
) -> gpd.GeoDataFrame:
    """
    Loads the hydrofabric and prepares the index for joining with model output.
    """
    try:
        gdf = gpd.read_file(gpkg_path, layer=layer)
    except ValueError:
        # Fallback if specific layer name is wrong
        gdf = gpd.read_file(gpkg_path)
    
    # Ensure we define the ID column cleanly
    col_to_use = id_col
    if id_col not in gdf.columns:
        if 'id' in gdf.columns: col_to_use = 'id'
        elif 'comid' in gdf.columns: col_to_use = 'comid'
        else:
            # Fallback to index if no ID found
            gdf.index.name = 'feature_id'
            return gdf

    # Create a clean ID for joining
    if match_netcdf_ids:
        # Regex to remove non-numeric prefixes (wb-, nex-, cat-)
        # We explicitly handle the case where extraction fails (returns NaN)
        extracted = gdf[col_to_use].astype(str).str.extract(r'(\d+)')
        
        # Drop rows where ID couldn't be parsed
        if extracted.isna().all().all():
             print("Warning: Could not parse integer IDs from hydrofabric. Keeping original strings.")
             gdf['feature_id'] = gdf[col_to_use]
        else:
             gdf['feature_id'] = extracted.fillna(0).astype(int)
    else:
        gdf['feature_id'] = gdf[col_to_use]

    return gdf.set_index('feature_id')

def extract_usgs_mapping(hydrofabric: gpd.GeoDataFrame, gage_col: str = "gage_id") -> dict:
    """
    Creates a mapping dictionary from Feature ID -> USGS Site ID.
    
    Args:
        hydrofabric: Loaded GeoDataFrame.
        gage_col: Column name containing USGS IDs (e.g., 'gage_id' or 'poi_id').
    
    Returns:
        dict: {feature_id: usgs_site_id}
    """
    if gage_col not in hydrofabric.columns:
        raise ValueError(f"Column '{gage_col}' not found in hydrofabric.")
        
    # Filter rows that actually have a gage
    subset = hydrofabric[hydrofabric[gage_col].notna()]
    
    # Create dict
    mapping = pd.Series(subset[gage_col].values, index=subset.index).to_dict()
    
    return mapping

def save_ensemble_stats(ds: xr.Dataset, output_path: str):
    """
    Saves the processed statistics to a NetCDF file.
    """
    print(f"Saving ensemble statistics to {output_path}...")
    
    # Compress the output to save space
    encoding = {var: {'zlib': True, 'complevel': 5} for var in ds.data_vars}
    
    ds.to_netcdf(output_path, encoding=encoding)
    print("✅ Save complete.")