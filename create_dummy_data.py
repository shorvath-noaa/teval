import xarray as xr
import pandas as pd
import numpy as np
import geopandas as gpd
from pathlib import Path
import sys
import argparse

# Ensure we can import teval modules
sys.path.append("src")
from teval.obs import usgs
import teval.utils as tutils

def create_realistic_output_from_gpkg(
    output_dir: str,
    gpkg_path: str,
    num_members: int = 5,
    start_date: str = "2023-05-01",
    end_date: str = "2023-05-30"
):
    """
    Generates synthetic t-route output files derived from a hydrofabric GeoPackage.
    """
    gpkg_file = Path(gpkg_path)
    if not gpkg_file.exists():
        raise FileNotFoundError(f"GeoPackage not found: {gpkg_path}")

    print(f"--- Generating Realistic Dummy Data from {gpkg_file.name} ---")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Read Hydrofabric
    print(f"1. Reading GeoPackage: {gpkg_path}...")
    try:
        gdf = gpd.read_file(gpkg_path, layer="flowpath-attributes")
    except ValueError:
        gdf = gpd.read_file(gpkg_path, layer="flowpaths")

    # Extract all feature IDs for the domain
    feature_ids = gdf['id'].astype(str).str.extract(r'(\d+)').astype("int32")[0].values
    print(f"   Found {len(feature_ids)} features in domain.")
    
    # 2. Identify the Gage
    gage_id = None
    primary_feature_id = None

    # Attempt 1: Look in the 'gage' column
    if 'gage' in gdf.columns:
        gages_found = gdf[gdf['gage'].notna() & (gdf['gage'] != "")]
        if not gages_found.empty:
            gage_id = gages_found.iloc[0]['gage']
            primary_feature_id = gages_found.iloc[0]['id']
            print(f"   Detected Gage ID from attributes: {gage_id}")

    # Attempt 2: Fallback to filename extraction
    if gage_id is None:
        print("   No gage ID found in attributes. Checking filename...")
        import re
        match = re.search(r"gage_(\d+)", gpkg_file.name)
        if match:
            gage_id = match.group(1)
            # Find a valid feature ID to attach to
            primary_feature_id_idx = tutils.find_tailwater_feature(gdf) 
            primary_feature_id = gdf.loc[primary_feature_id_idx]['id'].iloc[0]
            print(f"   Detected Gage ID from filename: {gage_id}")
        else:
            raise ValueError(f"Could not determine Gage ID from attributes or filename ({gpkg_file.name}).")
    
    primary_feature_id = int(str(primary_feature_id).replace("wb-", ""))
    
    # 3. Fetch Real USGS Data
    print(f"2. Fetching USGS data for gage {gage_id}...")
    df_obs = usgs.fetch_usgs_streamflow(
        [gage_id], start_date, end_date, to_cms=True, to_utc=True
    )
    
    if df_obs.empty:
        print("   Warning: Could not fetch data. Generating pure random noise.")
        times = pd.date_range(start_date, end_date, freq="h")
        truth_flow = np.random.gamma(shape=2.5, scale=1.0, size=len(times))
    else:
        df_obs = df_obs.resample("1h").mean().interpolate()
        times = df_obs.index.tz_localize(None)
        times.name = "time"
        truth_flow = df_obs.iloc[:, 0].values.astype("float32")
    
    num_timesteps = len(times)
    num_features = len(feature_ids)
    
    # 4. Generate Ensemble Members
    print(f"3. Generating {num_members} ensemble files in {output_dir}...")
    
    for i in range(num_members):
        data_flow = np.zeros((num_timesteps, num_features), dtype="float32")
        data_vel = np.zeros((num_timesteps, num_features), dtype="float32")
        data_depth = np.zeros((num_timesteps, num_features), dtype="float32")
        
        # Bias logic
        bias = (i - num_members // 2) * 0.10
        gamma_noise = np.random.gamma(10.0, 0.1, size=num_timesteps)
        simulated_flow = truth_flow * (1 + bias) + gamma_noise
        
        # Distribute logic
        primary_idx = np.where(feature_ids == int(primary_feature_id))[0][0]
        data_flow[:, primary_idx] = simulated_flow
        
        for f_idx in range(num_features):
            if f_idx == primary_idx: continue
            scale = (feature_ids[f_idx] % 10) / 10.0  
            data_flow[:, f_idx] = simulated_flow * scale
            
        data_vel = (data_flow ** 0.4).astype("float32")
        data_depth = (data_flow ** 0.4).astype("float32")

        ds = xr.Dataset(
            data_vars={
                "streamflow": (("time", "feature_id"), data_flow),
                "velocity":   (("time", "feature_id"), data_vel),
                "depth":      (("time", "feature_id"), data_depth)
            },
            coords={
                "time": times,
                "feature_id": feature_ids
            },
            attrs={"Formulation_ID": i + 1}
        )

        file_name = f"troute_output_formulation_{i+1}.nc"
        ds.to_netcdf(output_dir / file_name)

    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate dummy ensemble data from a hydrofabric.")
    parser.add_argument(
        "gpkg", 
        type=str, 
        help="Path to the input hydrofabric GeoPackage (.gpkg)"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="data", 
        help="Directory to save the generated NetCDF files (default: data)"
    )
    parser.add_argument(
        "--members", 
        type=int, 
        default=10, 
        help="Number of ensemble members to generate (default: 10)"
    )

    args = parser.parse_args()

    create_realistic_output_from_gpkg(
        output_dir=args.output,
        gpkg_path=args.gpkg,
        num_members=args.members
    )