import pandas as pd
import xarray as xr
from pathlib import Path
import time
import logging
from typing import Optional

# MUST BE SET BEFORE IMPORTING PYPLOT to ensure process safety in Joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import geopandas as gpd
import numpy as np
import contextily as cx


def hydrograph(
    stats_ds: xr.Dataset, 
    feature_id: int, 
    var_name: str = "streamflow", 
    ax=None, 
    obs_series=None,
    plot_uncertainty: bool = True,
    plot_members: bool = False,
    ensemble_ds: xr.Dataset = None,
    quantiles: list = [0.05, 0.95],
    metrics_df: Optional[pd.DataFrame] = None # <-- NEW ARGUMENT
):
    """
    Plots mean, median, and uncertainty bounds (or individual members) with optional metrics.
    """
    if ax is None:
        ax = plt.gca()
        
    # Helper to generate a clean string of metrics for the legend (e.g. "(NSE: 0.85, KGE: 0.72)")
    def get_metrics_str(source_name: str) -> str:
        if metrics_df is None or metrics_df.empty: return ""
        
        # Find the row for this specific model/source
        row = metrics_df[metrics_df['source'].astype(str) == source_name]
        if row.empty: return ""
        
        skip_cols = {'feature_id', 'gage_id', 'lat', 'lon', 'source', 'sig_class'}
        parts = []
        for col in row.columns:
            if col not in skip_cols and pd.notnull(row[col].iloc[0]):
                val = row[col].iloc[0]
                if isinstance(val, (int, float)):
                    parts.append(f"{col.upper()}: {val:.2f}")
                    
        return f" ({', '.join(parts)})" if parts else ""

    # 1. Select data
    try:
        data = stats_ds.sel(feature_id=feature_id)
    except KeyError:
        print(f"Error: Feature ID {feature_id} not found in dataset.")
        return
    
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

    qs = sorted(quantiles)
    if len(qs) >= 2:
        q_lower, q_upper = qs[0], qs[-1]
    else:
        q_lower, q_upper = 0.05, 0.95 
    
    def fmt_q(q): return f"p{int(q*100):02d}"
    
    lbl_lower, lbl_upper = fmt_q(q_lower), fmt_q(q_upper)
    
    p_lower = get_flat(lbl_lower, mean)
    p_upper = get_flat(lbl_upper, mean)

    if pd.api.types.is_datetime64_any_dtype(data.time):
        times = data.time.values
    elif 'reference_time' in data.coords:
        ref_time = pd.to_datetime(data.reference_time.values)
        hours = pd.to_timedelta(data.time.values, unit='h')
        times = ref_time + hours
    else:
        times = data.time.values
    
    # 4. Plot Individual Members (Spaghetti) WITH METRICS
    if plot_members:
        if ensemble_ds is not None:
            try:
                member_data = ensemble_ds[var_name].sel(feature_id=feature_id)
                member_dim = None
                dims_map = {d.lower(): d for d in member_data.dims}
                for t in ['formulation_id', 'member', 'ensemble', 'run', 'formulation']:
                    if t in dims_map:
                        member_dim = dims_map[t]
                        break
                
                if member_dim:
                    n_members = member_data.sizes[member_dim]
                    if n_members <= 20:
                        colors = cm.tab20(np.linspace(0, 1, n_members))
                    else:
                        colors = cm.jet(np.linspace(0, 1, n_members))

                    for i in range(n_members):
                        trace = member_data.isel({member_dim: i}).values.flatten()
                        try:
                            # E.g., 'model_A'
                            mid = str(member_data[member_dim].values[i])
                            # Pull metrics specifically for 'model_A'
                            lbl = f"{mid}{get_metrics_str(mid)}"
                        except:
                            lbl = f"Member {i}"

                        ax.plot(times, trace, color=colors[i], alpha=0.7, 
                                linewidth=1.0, label=lbl, zorder=1)
                else:
                    print(f"Warning: Could not identify member dimension.")
            except Exception as e:
                print(f"Warning: Could not plot members: {e}")

    # 5. Plot Uncertainty Band
    if plot_uncertainty and not plot_members:
        pct = int(round(q_upper - q_lower, 2) * 100)
        band_label = f"{pct}% Uncertainty ({lbl_lower}-{lbl_upper})"
        ax.fill_between(times, p_lower, p_upper, color='gray', alpha=0.3, 
                        label=band_label, zorder=2)
    
    # 6. Central Tendencies WITH METRICS
    mean_label = f"Ensemble Mean{get_metrics_str('ensemble_mean')}"
    ax.plot(times, mean, 'k-', linewidth=2.5, label=mean_label, zorder=4)
    
    if median is not None:
        ax.plot(times, median, 'b--', linewidth=2.0, label='Ensemble Median', zorder=4)
    
    # 7. Observations
    if obs_series is not None:
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
    
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    
    # Allow the legend to wrap nicely if it gets wide with metrics
    ax.legend(by_label.values(), by_label.keys(), loc='upper center', bbox_to_anchor=(0.5, -0.15), fontsize='small', ncol=2)
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
    """
    if 'lat' not in metrics_df.columns or 'lon' not in metrics_df.columns:
        raise ValueError("CSV missing 'lat' or 'lon' columns. Cannot plot map.")
    
    df = metrics_df.dropna(subset=['lat', 'lon', variable])
    
    if df.empty:
        print(f"No valid data found for variable '{variable}'. Skipping map.")
        return

    gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df.lon, df.lat), crs="EPSG:4326"
    )
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    if variable == "sig_class":
        color_map = {
            'skillful': 'blue', 
            'unskillful': 'red', 
            'indeterminate': 'gray',
            'insufficient_data': 'lightgray'
        }
        for cat, color in color_map.items():
            subset = gdf[gdf[variable] == cat]
            if not subset.empty:
                ax.scatter(
                    subset.geometry.x, subset.geometry.y, 
                    c=color, label=cat.title(), s=marker_size, edgecolors='k', linewidth=0.5, zorder=5
                )
        ax.legend(title="Hypothesis Test")
        
    else:
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

    if add_basemap:
        if gdf.crs.to_string() != "EPSG:3857":
            gdf = gdf.to_crs(epsg=3857)
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
    """
    if gdf.empty or metric_name not in gdf.columns:
        print(f"Skipping map for {metric_name}: No data found.")
        return

    fig, ax = plt.subplots(figsize=(12, 10))
    
    if gdf.crs is not None and gdf.crs.to_string() != "EPSG:3857":
        gdf = gdf.to_crs(epsg=3857)
    elif gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326).to_crs(epsg=3857)

    vmin, vmax = None, None
    if metric_name.lower() in ['nse', 'kge', 'kge_2012']:
        vmin, vmax = -1, 1
    
    markers = {
        'skillful': 'o',      
        'unskillful': 'X',    
        'indeterminate': 's', 
        'insufficient_data': 'v' 
    }
    
    unique_types = gdf[sig_column].unique()
    
    sc = None
    for sig_type in unique_types:
        subset = gdf[gdf[sig_column] == sig_type]
        if subset.empty: continue
        
        marker = markers.get(sig_type, 'o')
        edge_color = 'black'
        line_width = 1.0
        
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

    if sc:
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.1)
        cbar = plt.colorbar(sc, cax=cax)
        cbar.set_label(metric_name.upper(), fontsize=12)

    legend_handles = []
    for sig_type, marker in markers.items():
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

    try:
        cx.add_basemap(ax, source=cx.providers.CartoDB.Positron)
    except Exception:
        pass
        
    ax.set_axis_off()
    ax.set_title(f"Metric: {metric_name.upper()} & Significance", fontsize=15)
    
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    print(f"Map saved to {output_path}")
    plt.close(fig)