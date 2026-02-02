import pandas as pd
import matplotlib.pyplot as plt
import teval.io as tio
import teval.stats as tstats
import teval.viz.static as tviz
import teval.viz.interactive as tinteractive
import teval.obs.usgs as tobs
import teval.metrics.deterministic as tmetrics
import teval.utils as tutils
from pathlib import Path

def main():
    print("--- TEVAL Example Runner ---")
    
    # 1. Setup Paths
    data_dir = Path("data")
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # 2. Load Data
    print("1. Loading Troute Output Data...")
    try:
        # Matches any .nc file in data/
        ds_ensemble = tio.load_ensemble(str(data_dir / "*.nc"))
        print(f"   Loaded {len(ds_ensemble.Formulation_ID)} members.")
    except Exception as e:
        print(f"   Error loading data: {e}")
        return

    # 3. Load Hydrofabric (Needed for Gage Lookup)
    print("2. Loading Hydrofabric (for Gage lookup)...")
    gdf_hydro = None
    try:
        gpkg_files = list(data_dir.glob("*.gpkg"))
        if gpkg_files:
            gdf_hydro = tio.load_hydrofabric(gpkg_files[0], layer='flowpath-attributes')
            gdf_hydro_geom = tio.load_hydrofabric(gpkg_files[0], layer='flowpaths')
            print(f"   Loaded hydrofabric: {gpkg_files[0].name}")
        else:
            print("   No .gpkg file found. Skipping observation lookup.")
    except Exception as e:
        print(f"   Warning: Could not load hydrofabric ({e})")
        
    # 4. Calculate Statistics
    print("3. Calculating Statistics...")
    ds_stats = tstats.calculate_basics(ds_ensemble)
    print("   Stats calculated: Mean, Median, Std, p05, p95")
    
    # 5. Save Results
    output_file = output_dir / "ensemble_stats.nc"
    print(f"4. Saving NetCDF to {output_file}...")
    tio.save_ensemble_stats(ds_stats, output_file)

    # 6. Generate Plot
    print("5. Generating Hydrograph...")
    # Pick the tailwater feature_id found in the data
    target_id = tutils.find_tailwater_feature(gdf_hydro)
    
    # Retrieve gage data if available
    gage_id = gdf_hydro.loc[target_id].gage.values[0]
    if gage_id:        
        # Download Data (align dates with model)
        start_date = str(pd.to_datetime(ds_stats.time.values[0]).date())
        end_date = str(pd.to_datetime(ds_stats.time.values[-1]).date())
        
        obs_df = tobs.fetch_usgs_streamflow(
            gage_id, start_date, end_date
        )
        
        obs_series = None
        if not obs_df.empty:
            # obs_df = obs_df.iloc[:, 0] # Take the first column
            
            # Calculate Metrics
            # Extract model mean for comparison
            sim_df = ds_stats['streamflow_mean'].sel(feature_id=target_id).to_dataframe()
            scores = tmetrics.calculate_all(sim_df, obs_df)
            
            print("   Performance Metrics:")
            for k, v in scores.items():
                print(f"     {k}: {v:.3f}")
            
            # Save obs for plotting later
            obs_series = obs_df.iloc[:,0]
        else:
            print("   No data available from USGS for this period.")
    else:
        print("   No USGS gage found nearby.")
    
    # Plot Hydrograph
    fig, ax = plt.subplots(figsize=(10, 6))
    tviz.hydrograph(ds_stats, feature_id=target_id, ax=ax, obs_series=obs_series)
    
    plot_file = output_dir / f"hydrograph_{target_id}.png"
    plt.savefig(plot_file)
    print(f"   Hydrograph saved.")
    
    # Plot Map
    if gdf_hydro_geom is not None:
        print("6. Generating Map...")
        fig, ax = plt.subplots(figsize=(10, 10))
        gpkg_name = gpkg_files[0].with_suffix('').name
        
        # Map the results
        tviz.map_network(
            gdf_hydro_geom, 
            ds_stats, 
            var_name="streamflow_mean", 
            time_index=-1, # Last timestep
            source_name=gpkg_name,
            add_basemap=True,
            basemap_provider="USGSTopo",
            ax=ax
        )
        
        map_file = output_dir / "map_streamflow_mean.png"
        plt.savefig(map_file)
        print(f"   Map saved to {map_file}")
    
    # 7. Generate Interactive Map
    if gdf_hydro_geom is not None:
        print("7. Generating Interactive Map...")
        html_file = output_dir / "map_interactive.html"
        
        try:
            tinteractive.map_folium(
                gdf_hydro_geom,
                ds_stats, 
                var_name="streamflow_mean", 
                time_index=-1, # Last timestep
                output_html=str(html_file)
            )
        except Exception as e:
            print(f"   Could not generate interactive map: {e}")
    
    print("\n✅ Example run completed successfully.")

if __name__ == "__main__":
    main()