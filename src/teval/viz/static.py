import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
import geopandas as gpd

def hydrograph(stats_ds: xr.Dataset, feature_id: int, var_name: str = "streamflow", ax=None, obs_series=None):
    """
    Plots mean, median, and 5-95% uncertainty bounds on a given axes.
    """
    if ax is None:
        ax = plt.gca()
        
    # 1. Select data for the specific feature
    try:
        data = stats_ds.sel(feature_id=feature_id)
    except KeyError:
        print(f"Error: Feature ID {feature_id} not found in dataset.")
        return

    # 2. Extract calculated stats variables
    # Check if specific var exists (e.g. streamflow_mean) or if the user passed the suffix already
    if f"{var_name}_mean" in data:
        mean = data[f"{var_name}_mean"]
        median = data[f"{var_name}_median"]
        p05 = data[f"{var_name}_p05"]
        p95 = data[f"{var_name}_p95"]
    else:
        # Fallback: maybe the user passed a dataset that is already just streamflow stats?
        # This makes it safer if names vary
        mean = data["mean"] if "mean" in data else data
        median = data["median"] if "median" in data else data
        
    # 3. Handle Time Index
    # Check if the time is already a datetime object
    if pd.api.types.is_datetime64_any_dtype(data.time):
        times = data.time.values
    elif 'reference_time' in data.coords:
        # Only do the offset math if time is not already datetime
        ref_time = pd.to_datetime(data.reference_time.values)
        hours = pd.to_timedelta(data.time.values, unit='h')
        times = ref_time + hours
    else:
        times = data.time.values

    # 4. Plotting
    # Uncertainty Band (5th to 95th percentile)
    ax.fill_between(times, p05, p95, color='gray', alpha=0.3, label='90% Uncertainty (p05-p95)')
    
    # Central Tendencies
    ax.plot(times, mean, 'b-', linewidth=2, label='Ensemble Mean')
    ax.plot(times, median, 'b--', linewidth=1, label='Ensemble Median')
    
    # Optional: Observations
    if obs_series is not None:
        # Slice obs to match plot range
        try:
            obs_sub = obs_series.loc[times[0]:times[-1]]
            ax.plot(obs_sub.index, obs_sub.values, 'k.', markersize=4, label='Observations')
        except:
            print("Warning: Could not align observations with plot times.")
    
    # Formatting
    ax.set_title(f"Ensemble Hydrograph: Feature {feature_id}")
    ax.set_ylabel(f"{var_name}")
    ax.set_xlabel("Time")
    ax.legend()
    ax.grid(True, alpha=0.3)

def map_network(gdf: gpd.GeoDataFrame, stats_ds: xr.Dataset, var_name: str = "streamflow_mean", time_index: int = -1, ax=None):
    """
    Plots a static choropleth map of the river network.
    """
    if ax is None:
        ax = plt.gca()

    # 1. Select specific time slice
    data_slice = stats_ds.isel(time=time_index)
    
    # 2. Convert to DataFrame for merging
    df_data = data_slice[var_name].to_dataframe().reset_index()
    
    # 3. Merge with Geometry
    # Ensure feature_id is available for merge
    gdf_plot = gdf.copy()
    if 'feature_id' not in gdf_plot.columns:
        gdf_plot = gdf_plot.reset_index() 
        
    map_data = gdf_plot.merge(df_data, on='feature_id', how='inner')
    
    # 4. Plot
    map_data.plot(
        column=var_name,
        ax=ax,
        legend=True,
        cmap='viridis',
        linewidth=2,
        legend_kwds={'label': var_name, 'shrink': 0.6}
    )
    
    ax.set_axis_off()
    ax.set_title(f"Map: {var_name} (Time Step {time_index})")