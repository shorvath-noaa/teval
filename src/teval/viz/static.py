import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import xarray as xr
import pandas as pd
import geopandas as gpd
import contextily as cx

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
        mean = data[f"{var_name}_mean"].values.flatten()
        median = data[f"{var_name}_median"].values.flatten()
        p05 = data[f"{var_name}_p05"].values.flatten()
        p95 = data[f"{var_name}_p95"].values.flatten()
    else:
        # Fallback
        mean = data["mean"].values.flatten() if "mean" in data else data.values.flatten()
        median = data["median"].values.flatten() if "median" in data else data.values.flatten()
        # Handle case where p05/p95 might not exist (e.g. single member)
        p05 = data["p05"].values.flatten() if "p05" in data else mean
        p95 = data["p95"].values.flatten() if "p95" in data else mean
        
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
        # Check if plot times are tz-aware (e.g. UTC) vs naive
        plot_tz = None
        if hasattr(times, 'tz'): plot_tz = times.tz
        elif hasattr(times, 'dtype') and hasattr(times.dtype, 'tz'): plot_tz = times.dtype.tz
        elif len(times) > 0 and hasattr(times[0], 'tzinfo'): plot_tz = times[0].tzinfo
        
        obs_tz = obs_series.index.tz
        
        # If mismatch, convert obs to match plot
        if obs_tz is not None and plot_tz is None:
            obs_series = obs_series.tz_convert(None)
        elif obs_tz is None and plot_tz is not None:
            obs_series = obs_series.tz_localize(plot_tz)

        # Slice obs to match plot range
        try:
            # Use string slicing for robustness
            start_str = str(pd.to_datetime(times[0]))
            end_str = str(pd.to_datetime(times[-1]))
            obs_sub = obs_series.loc[start_str:end_str]
            
            if not obs_sub.empty:
                ax.plot(obs_sub.index, obs_sub.values, 'k.', markersize=4, label='Observations', zorder=10)
        except Exception as e:
            print(f"Warning: Could not align observations with plot times: {e}")
    
    # Formatting
    ax.set_title(f"Ensemble Hydrograph: Feature {feature_id}")
    ax.set_ylabel(f"{var_name}")
    ax.set_xlabel("Time")
    ax.legend()
    ax.grid(True, alpha=0.3)

def map_network(gdf: gpd.GeoDataFrame, 
                stats_ds: xr.Dataset, 
                var_name: str = "streamflow_mean", 
                time_index: int = -1, 
                source_name: str = "Network",
                add_basemap: bool = True,
                basemap_provider: str= "USGSTopo",
                ax=None):
    """
    Plots a choropleth map with optional baselayer.
    
    Args:
        gdf: GeoDataFrame of the river network.
        stats_ds: Xarray Dataset containing the stats.
        var_name: Variable to plot (e.g., 'streamflow_mean').
        time_index: Time index to plot (-1 for last step).
        source_name: Name for the title (e.g. filename).
        add_basemap: Whether to download and add a background map.
        basemap_provider: Contextily provider (e.g., cx.providers.OpenTopoMap). 
                          If None, defaults to OpenTopoMap.
        ax: Matplotlib axes.
    """
    if ax is None:
        ax = plt.gca()

    # 1. Select specific time slice
    data_slice = stats_ds.isel(time=time_index)
    # Extract Timestamp string
    raw_time = data_slice.time.values
    time_str = str(pd.to_datetime(raw_time)).replace("T", " ")
        
    # 2. Convert to DataFrame for merging
    df_data = data_slice[var_name].to_dataframe().reset_index()
    
    # 3. Merge with Geometry
    # Ensure feature_id is available for merge
    gdf_plot = gdf.copy()
    if 'feature_id' not in gdf_plot.columns:
        gdf_plot = gdf_plot.reset_index() 
        
    map_data = gdf_plot.merge(df_data, on='feature_id', how='inner')
    
    # Set custom color map
    hydro_cmap = mcolors.LinearSegmentedColormap.from_list(
        "hydro_flow", 
        ["white", "lightskyblue", "skyblue", "deepskyblue", "dodgerblue", "blue"]
    )
    
    # 4. Plot
    map_data.plot(
        column=var_name,
        ax=ax,
        legend=True,
        cmap=hydro_cmap,
        linewidth=3,
        alpha=0.9,
        legend_kwds={'label': "Streamflow (cms)", 'shrink': 0.7}
    )
    
    if add_basemap:
        basemap_dict = {
            "OpenTopoMap": cx.providers.OpenTopoMap,
            "CartoDBPositron": cx.providers.CartoDB.Positron,
            "USGSTopo": cx.providers.USGS.USTopo
        }
        if basemap_provider is None:
            # Default to USGSTopoMap if not specified
            basemap_provider = cx.providers.USGS.USTopo 
        cx.add_basemap(ax, crs=map_data.crs, source=basemap_dict.get(basemap_provider, cx.providers.OpenTopoMap))
    
    ax.set_axis_off()
    ax.set_title(f"{source_name} | {time_str} | {var_name}", fontsize=12)