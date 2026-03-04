import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import xarray as xr
import pandas as pd
import geopandas as gpd
import numpy as np
import contextily as cx
from typing import Optional
from pathlib import Path

def hydrograph(
    stats_ds: xr.Dataset, 
    feature_id: int, 
    var_name: str = "streamflow", 
    ax=None, 
    obs_series=None,
    plot_uncertainty: bool = True,
    plot_members: bool = False,
    ensemble_ds: xr.Dataset = None,
    quantiles: list = [0.05, 0.95]
):
    """
    Plots mean, median, and uncertainty bounds (or individual members).
    """
    if ax is None:
        ax = plt.gca()
        
    # 1. Select data
    try:
        data = stats_ds.sel(feature_id=feature_id)
    except KeyError:
        print(f"Error: Feature ID {feature_id} not found in dataset.")
        return
    
    # 2. Extract Stats
    # Helper to retrieve data safely
    def get_flat(key, default=None):
        if f"{var_name}_{key}" in data:
            return data[f"{var_name}_{key}"].values.flatten()
        elif key in data:
            return data[key].values.flatten()
        return default

    mean = get_flat("mean")
    median = get_flat("median")
    
    if mean is None:
        print(f"Warning: Mean statistic not found for {feature_id}. Skipping plot.")
        return

    # Determine Quantile Bounds dynamically
    qs = sorted(quantiles)
    if len(qs) >= 2:
        q_lower, q_upper = qs[0], qs[-1]
    else:
        q_lower, q_upper = 0.05, 0.95 # Fallback
    
    # Format strings to match variable names (e.g. 0.05 -> "p05")
    def fmt_q(q): return f"p{int(q*100):02d}"
    
    lbl_lower, lbl_upper = fmt_q(q_lower), fmt_q(q_upper)
    
    p_lower = get_flat(lbl_lower, mean) # Fallback to mean if missing
    p_upper = get_flat(lbl_upper, mean)

    # 3. Handle Time
    if pd.api.types.is_datetime64_any_dtype(data.time):
        times = data.time.values
    elif 'reference_time' in data.coords:
        ref_time = pd.to_datetime(data.reference_time.values)
        hours = pd.to_timedelta(data.time.values, unit='h')
        times = ref_time + hours
    else:
        times = data.time.values
    
    # 4. Plot Individual Members (Spaghetti)
    if plot_members:
        if ensemble_ds is not None:
            try:
                member_data = ensemble_ds[var_name].sel(feature_id=feature_id)
                
                # Case-Insensitive Dim Search
                member_dim = None
                dims_map = {d.lower(): d for d in member_data.dims}
                for t in ['formulation_id', 'member', 'ensemble', 'run']:
                    if t in dims_map:
                        member_dim = dims_map[t]
                        break
                
                if member_dim:
                    n_members = member_data.sizes[member_dim]
                    # Color generation
                    if n_members <= 20:
                        colors = cm.tab20(np.linspace(0, 1, n_members))
                    else:
                        colors = cm.jet(np.linspace(0, 1, n_members))

                    for i in range(n_members):
                        trace = member_data.isel({member_dim: i}).values.flatten()
                        try:
                            # Try to get actual Member ID
                            mid = member_data[member_dim].values[i]
                            lbl = f"Member {mid}"
                        except:
                            lbl = f"Member {i}"

                        ax.plot(times, trace, color=colors[i], alpha=0.7, 
                                linewidth=1.0, label=lbl, zorder=1)
                else:
                    print(f"Warning: Could not identify member dimension.")
            except Exception as e:
                print(f"Warning: Could not plot members: {e}")
        else:
            print("Warning: 'plot_members' is True but 'ensemble_ds' was not provided.")

    # 5. Plot Uncertainty Band
    if plot_uncertainty and not plot_members:
        # Calculate percentage coverage for legend
        pct = int(round(q_upper - q_lower, 2) * 100)
        band_label = f"{pct}% Uncertainty ({lbl_lower}-{lbl_upper})"
        
        ax.fill_between(times, p_lower, p_upper, color='gray', alpha=0.3, 
                        label=band_label, zorder=2)
    
    # 6. Central Tendencies
    ax.plot(times, mean, 'k-', linewidth=2.5, label='Ensemble Mean', zorder=4)
    
    if median is not None:
        ax.plot(times, median, 'b--', linewidth=2.0, label='Ensemble Median', zorder=4)
    
    # 7. Observations
    if obs_series is not None:
        # Timezone Alignment
        plot_tz = None
        if hasattr(times, 'tz'): plot_tz = times.tz
        elif hasattr(times, 'dtype') and hasattr(times.dtype, 'tz'): plot_tz = times.dtype.tz
        elif len(times) > 0 and hasattr(times[0], 'tzinfo'): plot_tz = times[0].tzinfo
        
        obs_tz = obs_series.index.tz
        if obs_tz is not None and plot_tz is None:
            obs_series = obs_series.tz_convert(None)
        elif obs_tz is None and plot_tz is not None:
            obs_series = obs_series.tz_localize(plot_tz)

        try:
            s, e = str(pd.to_datetime(times[0])), str(pd.to_datetime(times[-1]))
            obs_sub = obs_series.loc[s:e]
            if not obs_sub.empty:
                ax.plot(obs_sub.index, obs_sub.values, 'r.', markersize=8, 
                        label='Observations', zorder=10)
        except Exception:
            pass
            
    ax.set_title(f"Ensemble Hydrograph: Feature {feature_id}")
    ax.set_ylabel(var_name)
    ax.set_xlabel("Time")
    
    # Legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize='small', ncol=2)
    
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
    Plots a choropleth map.
    """
    if ax is None:
        ax = plt.gca()

    data_slice = stats_ds.isel(time=time_index)
    raw_time = data_slice.time.values
    time_str = str(pd.to_datetime(raw_time)).replace("T", " ")
        
    df_data = data_slice[var_name].to_dataframe().reset_index()
    
    gdf_plot = gdf.copy()
    gdf_plot = gdf_plot.reset_index().rename(columns={"id": "feature_id"})

    map_data = gdf_plot.merge(df_data, on='feature_id', how='inner')
    
    if map_data.empty:
        print("Warning: Map merge resulted in empty dataset.")
        return
    
    hydro_cmap = mcolors.LinearSegmentedColormap.from_list(
        "hydro_flow", 
        ["lightskyblue", "skyblue", "deepskyblue", "dodgerblue", "blue"]
    )
    
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
            basemap_provider = cx.providers.USGS.USTopo 
        cx.add_basemap(ax, crs=map_data.crs, source=basemap_dict.get(basemap_provider, cx.providers.OpenTopoMap))
    
    ax.set_axis_off()
    ax.set_title(f"{source_name} | {time_str} | {var_name}", fontsize=12)

def map_metrics(
    metrics_df: pd.DataFrame, 
    variable: str = "nse", 
    output_path: Optional[Path] = None,
    add_basemap: bool = True,
    cmap: str = "RdYlBu",
    marker_size: int = 25,
    title: str = None
):
    """
    Plots a domain-wide scatter map from a metrics CSV containing lat/lon columns.
    
    Args:
        csv_path: Path to the aggregated metrics.csv.
        variable: Column name to visualize (e.g., 'nse', 'kge', 'sig_class').
        output_path: Where to save the PNG.
    """
    # Filter rows with no geometry
    if 'lat' not in metrics_df.columns or 'lon' not in metrics_df.columns:
        raise ValueError("CSV missing 'lat' or 'lon' columns. Cannot plot map.")
    
    df = metrics_df.dropna(subset=['lat', 'lon', variable])
    
    if df.empty:
        print(f"No valid data found for variable '{variable}'. Skipping map.")
        return

    # Convert to GeoDataFrame
    gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df.lon, df.lat), crs="EPSG:4326"
    )
    
    # Setup Plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Handle Categorical vs Continuous data
    if variable == "sig_class":
        # Custom color mapping for significance
        color_map = {
            'skillful': 'blue', 
            'unskillful': 'red', 
            'indeterminate': 'gray',
            'insufficient_data': 'lightgray'
        }
        # Plot each category manually for a nice legend
        for cat, color in color_map.items():
            subset = gdf[gdf[variable] == cat]
            if not subset.empty:
                ax.scatter(
                    subset.geometry.x, subset.geometry.y, 
                    c=color, label=cat.title(), s=marker_size, edgecolors='k', linewidth=0.5, zorder=5
                )
        ax.legend(title="Hypothesis Test")
        
    else:
        # Continuous Plot (NSE, KGE)
        gdf = gdf.to_crs(epsg=3857)
        gdf.plot(
            column=variable, 
            ax=ax, 
            cmap=cmap, 
            legend=True, 
            markersize=marker_size,
            edgecolor='k',
            linewidth=0.3,
            legend_kwds={'label': variable.upper(), 'shrink': 0.7},
            zorder=5
        )

    # Basemap
    if add_basemap:
        if gdf.crs.to_string() != "EPSG:3857":
            gdf = gdf.to_crs(epsg=3857)
            # Re-set bounds based on data
            ax.set_xlim(gdf.total_bounds[0], gdf.total_bounds[2])
            ax.set_ylim(gdf.total_bounds[1], gdf.total_bounds[3])
            
        cx.add_basemap(ax, source=cx.providers.CartoDB.Positron)

    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=14)
    else:
        ax.set_title(f"Domain Metric: {variable}", fontsize=14)

    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        print(f"Map saved to {output_path}")
        plt.close(fig)

def plot_metric_with_significance(
    gdf: gpd.GeoDataFrame,
    metric_name: str,
    sig_column: str,
    output_path: Path,
    cmap: str = "RdYlBu",
    marker_size: int = 60
):
    """
    Plots a map where:
    - Color = Metric Value (Continuous)
    - Shape/Edge = Significance Class (Categorical)
    
    Shapes:
    - Skillful: Circle (o)
    - Unskillful: X (X)
    - Indeterminate: Square (s)
    - Insufficient Data: Triangle Down (v)
    """
    if gdf.empty or metric_name not in gdf.columns:
        print(f"Skipping map for {metric_name}: No data found.")
        return

    # 1. Setup Plot
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Reproject for basemap
    if gdf.crs is not None and gdf.crs.to_string() != "EPSG:3857":
        gdf = gdf.to_crs(epsg=3857)
    elif gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326).to_crs(epsg=3857)

    # 2. Define Value Ranges (Centering for NSE/KGE)
    vmin, vmax = None, None
    if metric_name.lower() in ['nse', 'kge', 'kge_2012']:
        # Center colormap at 0, clamp at -1 to 1 for visual clarity
        vmin, vmax = -1, 1
    
    # 3. Define Significance Styles
    markers = {
        'skillful': 'o',      
        'unskillful': 'X',    
        'indeterminate': 's', 
        'insufficient_data': 'v' 
    }
    
    # 4. Plot Loop (One scatter call per marker type to handle shapes)
    unique_types = gdf[sig_column].unique()
    
    sc = None
    for sig_type in unique_types:
        subset = gdf[gdf[sig_column] == sig_type]
        if subset.empty: continue
        
        marker = markers.get(sig_type, 'o')
        
        # Skillful gets a bold edge, others get thinner/gray edge
        edge_color = 'black'
        line_width = 1.0
        
        # Plot
        sc = ax.scatter(
            subset.geometry.x, subset.geometry.y,
            c=subset[metric_name],
            cmap=cmap,
            vmin=vmin, vmax=vmax,
            s=marker_size,
            marker=marker,
            edgecolors=edge_color,
            linewidth=line_width,
            label=sig_type.title(), 
            zorder=5
        )

    # 5. Colorbar (Metric Value)
    if sc:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.1)
        cbar = plt.colorbar(sc, cax=cax)
        cbar.set_label(metric_name.upper(), fontsize=12)

    # 6. Legend (Significance Class)
    # Manually create handles for the shapes so they show up black/gray in legend
    legend_handles = []
    for sig_type, marker in markers.items():
        # Only add to legend if it exists in the data
        if sig_type in unique_types:
            h = plt.Line2D(
                [], [], color='white', marker=marker, 
                markerfacecolor='gray', markeredgecolor='black', 
                markersize=10, label=sig_type.title()
            )
            legend_handles.append(h)
            
    if legend_handles:
        ax.legend(
            handles=legend_handles, 
            loc='lower left', 
            title="Hypothesis Test",
            frameon=True
        )

    # 7. Basemap & Polish
    try:
        cx.add_basemap(ax, source=cx.providers.CartoDB.Positron)
    except Exception:
        pass
        
    ax.set_axis_off()
    ax.set_title(f"Metric: {metric_name.upper()} & Significance", fontsize=15)
    
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    print(f"Map saved to {output_path}")
    plt.close(fig)