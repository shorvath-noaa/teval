"""
Static visualization functions for teval.

Functions
---------
hydrograph               : time-series plot of ensemble mean/band/members + obs
map_metrics              : gage scatter map coloured by a single metric
plot_metric_with_significance : metric value (colour) + hypothesis class (marker shape)

Skill map functions (from plot_more_metrics integration)
---------
get_best_models          : find the best-performing source per gage for a metric
plot_winner_map          : map of winning source per gage coloured by model identity
plot_boxplots            : per-formulation score distribution boxplots
plot_vpu_breakdown       : stacked-bar win-rate by VPU (CONUS-scale)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import contextily as cx

try:
    import seaborn as sns
    _HAS_SNS = True
except ImportError:
    _HAS_SNS = False

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hydrograph
# ---------------------------------------------------------------------------

def hydrograph(
    stats_ds: xr.Dataset,
    feature_id: int = None,
    gage_id: str = None,
    nexus_id: str = None,
    var_name: str = "streamflow",
    ax=None,
    obs_series=None,
    plot_uncertainty: bool = True,
    plot_members: bool = False,
    ensemble_ds: xr.Dataset = None,
    quantiles: list = None,
    metrics_df: Optional[pd.DataFrame] = None,
):
    """
    Plot ensemble mean/band (or spaghetti members) with optional observations.

    Parameters
    ----------
    stats_ds      : Dataset with dims (time[, feature_id]) containing
                    ``{var}_mean``, ``{var}_median``, ``{var}_p05``, ``{var}_p95``.
    feature_id    : Select a single feature from stats_ds. If None and
                    feature_id is a dimension, the first feature is used
                    automatically so that all arrays are 1-D.
    """
    if quantiles is None:
        quantiles = [0.05, 0.95]
    
    if ax is None:
        ax = plt.gca()

    # ------------------------------------------------------------------
    # Select a single feature so all downstream arrays are 1-D
    # ------------------------------------------------------------------
    if 'feature_id' in stats_ds.dims:
        if feature_id is not None:
            try:
                data = stats_ds.sel(feature_id=feature_id)
            except KeyError:
                logger.warning(f"Feature ID {feature_id} not found in dataset. Skipping.")
                return
        else:
            # Auto-select the first feature to guarantee 1-D output
            data = stats_ds.isel(feature_id=0)
    else:
        data = stats_ds

    def get_metrics_str(source_name: str) -> str:
        """Format a metrics dict into a short display string for plot annotations."""
        if metrics_df is None or metrics_df.empty:
            return ""
        row = metrics_df[metrics_df['source'].astype(str) == source_name]
        if row.empty:
            return ""
        skip_cols = {'feature_id', 'gage_id', 'lat', 'lon', 'source', 'sig_class'}
        parts = []
        for col in row.columns:
            if col not in skip_cols and pd.notnull(row[col].iloc[0]):
                val = row[col].iloc[0]
                if isinstance(val, (int, float, np.floating, np.integer)):
                    parts.append(f"{col.upper()}: {val:.2f}")
        return f" ({', '.join(parts)})" if parts else ""

    def get_flat(key, default=None):
        """Flatten a nested dict into a single-level dict."""
        full_key = f"{var_name}_{key}"
        if full_key in data:
            return data[full_key].values.flatten()
        elif key in data:
            return data[key].values.flatten()
        return default

    mean = get_flat("mean")
    median = get_flat("median")

    if mean is None:
        logger.warning(f"Mean statistic not found (feature_id={feature_id}). Skipping plot.")
        return

    qs = sorted(quantiles)
    q_lower, q_upper = (qs[0], qs[-1]) if len(qs) >= 2 else (0.05, 0.95)

    def fmt_q(q):
        """Format a quantile value as a percentage string."""
        return f"p{int(q * 100):02d}"

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

    # ------------------------------------------------------------------
    # Spaghetti members
    # ------------------------------------------------------------------
    if plot_members and ensemble_ds is not None:
        if 'feature_id' in ensemble_ds.dims:
            if feature_id is not None:
                member_data = ensemble_ds[var_name].sel(feature_id=feature_id)
            else:
                member_data = ensemble_ds[var_name].isel(feature_id=0)
        else:
            member_data = ensemble_ds[var_name]

        member_dim = None
        dims_map = {d.lower(): d for d in member_data.dims}
        for t in ['formulation_id', 'member', 'ensemble', 'run', 'formulation']:
            if t in dims_map:
                member_dim = dims_map[t]
                break

        if member_dim:
            n_members = member_data.sizes[member_dim]
            colors = (matplotlib.colormaps['tab20'](np.linspace(0, 1, n_members))
                      if n_members <= 20
                      else matplotlib.colormaps['jet'](np.linspace(0, 1, n_members)))

            for i in range(n_members):
                trace = member_data.isel({member_dim: i}).values.flatten()
                try:
                    mid = str(member_data[member_dim].values[i])
                    lbl = f"{mid}{get_metrics_str(mid)}"
                except Exception:
                    lbl = f"Member {i}"
                ax.plot(times, trace, color=colors[i], alpha=0.7, linewidth=1.0,
                        label=lbl, zorder=1)
        else:
            logger.warning("Could not identify member dimension for spaghetti plot.")

    # ------------------------------------------------------------------
    # Uncertainty band
    # ------------------------------------------------------------------
    if plot_uncertainty and not plot_members:
        pct = int(round(q_upper - q_lower, 2) * 100)
        band_label = f"{pct}% Uncertainty ({lbl_lower}–{lbl_upper})"
        ax.fill_between(times, p_lower, p_upper, color='gray', alpha=0.3,
                        label=band_label, zorder=2)

    # ------------------------------------------------------------------
    # Central tendencies
    # ------------------------------------------------------------------
    mean_label = f"Ensemble Mean{get_metrics_str('ensemble_mean')}"
    ax.plot(times, mean, 'k-', linewidth=2.5, label=mean_label, zorder=4)

    if median is not None:
        ax.plot(times, median, 'b--', linewidth=2.0, label='Ensemble Median', zorder=4)

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------
    if obs_series is not None:
        try:
            plot_tz = None
            if len(times) > 0 and hasattr(pd.to_datetime(times[0]), 'tzinfo'):
                plot_tz = pd.to_datetime(times[0]).tzinfo

            obs_tz = obs_series.index.tz
            if obs_tz is not None and plot_tz is None:
                obs_series = obs_series.tz_convert(None)
            elif obs_tz is None and plot_tz is not None:
                obs_series = obs_series.tz_localize(plot_tz)

            s = str(pd.to_datetime(times[0]))
            e = str(pd.to_datetime(times[-1]))
            obs_sub = obs_series.loc[s:e]
            if not obs_sub.empty:
                ax.plot(obs_sub.index, obs_sub.values, 'r.', markersize=8,
                        label='Observations', zorder=10)
        except Exception as e:
            logger.warning(f"Could not plot observations for gage '{gage_id}': {e}")
    # ------------------------------------------------------------------
    title_parts = []
    if gage_id:
        title_parts.append(f"Gage: {gage_id}")
    if nexus_id:
        title_parts.append(f"Nexus: {nexus_id}")
    if feature_id and not gage_id:
        title_parts.append(f"Feature: {feature_id}")
    ax.set_title(f"Ensemble Hydrograph: {' | '.join(title_parts)}")
    ax.set_ylabel(var_name)
    ax.set_xlabel("Time")

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(),
              loc='upper center', bbox_to_anchor=(0.5, -0.15),
              fontsize='small', ncol=2)
    ax.grid(True, alpha=0.3)


# ---------------------------------------------------------------------------
# Gage scatter map (metrics)
# ---------------------------------------------------------------------------

def map_metrics(
    metrics_df: pd.DataFrame,
    variable: str = "nse",
    output_path: Optional[Path] = None,
    add_basemap: bool = True,
    cmap: str = "RdYlBu",
    marker_size: int = 25,
    title: str = None,
):
    """Scatter map of a single performance metric across all gage locations."""
    if 'lat' not in metrics_df.columns or 'lon' not in metrics_df.columns:
        raise ValueError("DataFrame missing 'lat' or 'lon' columns.")

    df = metrics_df.dropna(subset=['lat', 'lon', variable])
    if df.empty:
        logger.warning(f"No valid data for variable '{variable}'. Skipping map.")
        return

    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat),
                            crs="EPSG:4326")

    fig, ax = plt.subplots(figsize=(12, 8))

    if variable == "sig_class":
        color_map = {
            'skillful': 'blue', 'unskillful': 'red',
            'indeterminate': 'gray', 'insufficient_data': 'lightgray',
        }
        for cat, color in color_map.items():
            subset = gdf[gdf[variable] == cat]
            if not subset.empty:
                ax.scatter(subset.geometry.x, subset.geometry.y,
                           c=color, label=cat.title(), s=marker_size,
                           edgecolors='k', linewidth=0.5, zorder=5)
        ax.legend(title="Hypothesis Test")
    else:
        gdf = gdf.to_crs(epsg=3857)
        vmin, vmax, extend_opt = None, None, 'neither'
        if variable.lower() in ['nse', 'kge']:
            vmin, vmax, extend_opt = 0.0, 1.0, 'min'
        elif variable.lower() == 'pbias':
            vmin, vmax, extend_opt = -100.0, 100.0, 'both'

        gdf.plot(column=variable, ax=ax, cmap=cmap, legend=True,
                 vmin=vmin, vmax=vmax, markersize=marker_size,
                 edgecolor='k', linewidth=0.3,
                 legend_kwds={'label': variable.upper(), 'shrink': 0.7,
                              'extend': extend_opt},
                 zorder=5)

    if add_basemap:
        try:
            if gdf.crs.to_string() != "EPSG:3857":
                gdf = gdf.to_crs(epsg=3857)
            cx.add_basemap(ax, source=cx.providers.CartoDB.Positron)
        except Exception as e:
            logger.warning(f"Basemap unavailable: {e}")

    ax.set_axis_off()
    ax.set_title(title or f"Domain Metric: {variable.upper()}", fontsize=14)

    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        logger.info(f"Map saved → {output_path}")
        plt.close(fig)


def plot_metric_with_significance(
    gdf: gpd.GeoDataFrame,
    metric_name: str,
    sig_column: str,
    output_path: Path,
    cmap: str = "RdYlBu",
    marker_size: int = 60,
):
    """Map where colour = metric value and marker shape = significance class."""
    if gdf.empty or metric_name not in gdf.columns:
        logger.warning(f"Skipping significance map for {metric_name}: no data.")
        return

    fig, ax = plt.subplots(figsize=(12, 10))

    if gdf.crs is not None and gdf.crs.to_string() != "EPSG:3857":
        gdf = gdf.to_crs(epsg=3857)
    elif gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326).to_crs(epsg=3857)

    vmin = -1 if metric_name.lower() in ['nse', 'kge', 'kge_2012'] else None
    vmax = 1  if metric_name.lower() in ['nse', 'kge', 'kge_2012'] else None

    markers = {
        'skillful': 'o', 'unskillful': 'X',
        'indeterminate': 's', 'insufficient_data': 'v',
    }
    unique_types = gdf[sig_column].unique()

    sc = None
    for sig_type in unique_types:
        subset = gdf[gdf[sig_column] == sig_type]
        if subset.empty:
            continue
        sc = ax.scatter(subset.geometry.x, subset.geometry.y,
                        c=subset[metric_name], cmap=cmap,
                        vmin=vmin, vmax=vmax, s=marker_size,
                        marker=markers.get(sig_type, 'o'),
                        edgecolors='black', linewidth=1.0,
                        label=sig_type.title(), zorder=5)

    if sc:
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.1)
        cbar = plt.colorbar(sc, cax=cax)
        cbar.set_label(metric_name.upper(), fontsize=12)

    legend_handles = [
        plt.Line2D([], [], color='white', marker=m,
                   markerfacecolor='gray', markeredgecolor='black',
                   markersize=10, label=t.title())
        for t, m in markers.items() if t in unique_types
    ]
    if legend_handles:
        ax.legend(handles=legend_handles, loc='lower left',
                  title="Hypothesis Test", frameon=True)

    try:
        cx.add_basemap(ax, source=cx.providers.CartoDB.Positron)
    except Exception:
        pass

    ax.set_axis_off()
    ax.set_title(f"Metric: {metric_name.upper()} & Significance", fontsize=15)
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    logger.info(f"Significance map saved → {output_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Skill maps  (integrated from plot_more_metrics.py)
# ---------------------------------------------------------------------------

def get_best_models(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """
    Return the best-performing source (model / ensemble_mean) at each gage.

    For NSE / KGE the best is the highest value.
    For PBIAS the best is the value closest to zero.

    Parameters
    ----------
    df     : combined metrics DataFrame (columns include metric, gage_id, source).
    metric : metric name (case-insensitive).

    Returns
    -------
    DataFrame with one row per gage containing the winning source.
    """
    df_clean = df.dropna(subset=[metric, 'gage_id', 'source']).copy()
    if df_clean.empty:
        return df_clean

    if metric.lower() == 'pbias':
        df_clean['_abs_metric'] = df_clean[metric].abs()
        idx = df_clean.groupby('gage_id')['_abs_metric'].idxmin()
        return df_clean.loc[idx].drop(columns='_abs_metric').copy()
    else:
        idx = df_clean.groupby('gage_id')[metric].idxmax()
        return df_clean.loc[idx].copy()


def plot_winner_map(
    metrics_df: pd.DataFrame,
    metric: str,
    output_path: Path,
    add_basemap: bool = True,
) -> None:
    """
    Scatter map coloured by which model/source performed best at each gage.

    Parameters
    ----------
    metrics_df  : combined metrics DataFrame with lat, lon, gage_id, source columns.
    metric      : which metric to use for determining the winner.
    output_path : where to save the PNG.
    add_basemap : whether to add a CartoDB.Positron basemap.
    """
    if 'lat' not in metrics_df.columns or 'lon' not in metrics_df.columns:
        logger.warning("plot_winner_map: missing lat/lon. Skipping.")
        return

    best_df = get_best_models(metrics_df, metric)
    if best_df.empty:
        logger.warning(f"plot_winner_map: no valid data for metric '{metric}'. Skipping.")
        return

    win_counts = best_df['source'].value_counts()
    best_df = best_df.copy()
    best_df['Legend'] = best_df['source'].apply(lambda x: f"{x} (n={win_counts[x]})")

    gdf = gpd.GeoDataFrame(best_df,
                           geometry=gpd.points_from_xy(best_df.lon, best_df.lat),
                           crs="EPSG:4326").to_crs(epsg=3857)

    fig, ax = plt.subplots(figsize=(12, 8))
    gdf.plot(column='Legend', ax=ax, cmap='Set1', legend=True,
             markersize=25, edgecolor='k', linewidth=0.3,
             legend_kwds={'title': 'Winning Source', 'loc': 'lower left'},
             zorder=5)

    if add_basemap:
        try:
            cx.add_basemap(ax, source=cx.providers.CartoDB.Positron)
        except Exception as e:
            logger.warning(f"Basemap unavailable: {e}")

    ax.set_axis_off()
    ax.set_title(f"Best Performing Model by Location: {metric.upper()}", fontsize=15)
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    logger.info(f"Winner map saved → {output_path}")
    plt.close(fig)


def plot_boxplots(
    metrics_df: pd.DataFrame,
    metric: str,
    output_path: Path,
) -> None:
    """
    Boxplot comparing the score distribution across formulations / sources.

    Parameters
    ----------
    metrics_df  : combined metrics DataFrame.
    metric      : metric column name.
    output_path : where to save the PNG.
    """
    plot_df = metrics_df.dropna(subset=[metric, 'source']).copy()

    # Clip extreme outliers so the body of the distribution is readable
    if metric.lower() in ['nse', 'kge']:
        plot_df = plot_df[plot_df[metric] > -2]
    elif metric.lower() == 'pbias':
        plot_df = plot_df[plot_df[metric].between(-150, 150)]

    if plot_df.empty:
        logger.warning(f"plot_boxplots: no valid data for '{metric}'. Skipping.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    if _HAS_SNS:
        sns.boxplot(data=plot_df, x='source', y=metric, ax=ax,
                    palette='Set2', hue='source', legend=False)
    else:
        # Fallback: plain matplotlib boxplot
        sources = sorted(plot_df['source'].unique())
        data_grouped = [plot_df.loc[plot_df['source'] == s, metric].values for s in sources]
        ax.boxplot(data_grouped, labels=sources)

    ideal = 0 if metric.lower() == 'pbias' else 1
    ax.axhline(ideal, color='red', linestyle='--', alpha=0.6, label='Ideal Score')
    ax.set_title(f"Overall {metric.upper()} Distribution by Model", fontsize=14)
    ax.set_ylabel(metric.upper())
    ax.set_xlabel("Formulation / Source")
    plt.xticks(rotation=45, ha='right')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    logger.info(f"Boxplot saved → {output_path}")
    plt.close(fig)


def plot_vpu_breakdown(
    metrics_df: pd.DataFrame,
    metric: str,
    output_path: Path,
    vpu_map: dict,
) -> None:
    """
    Stacked bar chart showing win-rate percentages per VPU region.

    Parameters
    ----------
    metrics_df  : combined metrics DataFrame.
    metric      : metric to use when determining the winner.
    output_path : where to save the PNG.
    vpu_map     : mapping of {gage_id (str, zero-padded) → vpu_id (str)}.
                  Build from the hydrofabric 'flowpath-attributes' layer;
                  pass an empty dict to skip this plot gracefully.
    """
    if not vpu_map:
        logger.info("plot_vpu_breakdown: no VPU map provided. Skipping.")
        return

    best_df = get_best_models(metrics_df, metric).copy()
    if best_df.empty:
        logger.warning(f"plot_vpu_breakdown: no valid data for '{metric}'. Skipping.")
        return

    best_df['vpuid'] = best_df['gage_id'].astype(str).map(vpu_map)
    plot_df = best_df.dropna(subset=['vpuid', 'source'])

    if plot_df.empty:
        logger.warning("plot_vpu_breakdown: no gages matched a VPU. Skipping.")
        return

    counts = plot_df.groupby(['vpuid', 'source']).size().unstack(fill_value=0)
    percentages = counts.div(counts.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(12, 6))
    percentages.plot(kind='bar', stacked=True, colormap='Set1', ax=ax,
                     edgecolor='black')
    ax.set_title(f"Winning Model Percentage by VPU ({metric.upper()})", fontsize=14)
    ax.set_ylabel("Percentage of Gages Won (%)")
    ax.set_xlabel("Vector Processing Unit (VPU)")
    ax.legend(title="Winning Source", bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    logger.info(f"VPU breakdown saved → {output_path}")
    plt.close(fig)


def build_vpu_map(hydrofabric_dir: Path) -> dict:
    """
    Build a gage_id -> vpu_id mapping by scanning the hydrofabric directory
    for a CONUS GeoPackage with a 'flowpath-attributes' layer.

    Returns an empty dict if no suitable file is found or the layer is absent.
    """
    # Look for a CONUS-scale gpkg (large file, name contains 'conus' or 'nextgen')
    candidates = (
        list(hydrofabric_dir.glob("*[Cc][Oo][Nn][Uu][Ss]*.gpkg"))
        + list(hydrofabric_dir.glob("*[Nn]ext[Gg]en*.gpkg"))
    )
    if not candidates:
        return {}

    gpkg_path = candidates[0]
    try:
        import fiona
        layers = fiona.listlayers(gpkg_path)
        if 'flowpath-attributes' not in layers:
            return {}
        fp_attr = gpd.read_file(gpkg_path, layer='flowpath-attributes')[['gage', 'vpuid']]
        fp_attr = fp_attr.dropna(subset=['gage', 'vpuid'])
        return dict(zip(fp_attr['gage'].astype(str).str.zfill(8),
                        fp_attr['vpuid'].astype(str)))
    except Exception as e:
        logger.warning(f"Could not build VPU map from {gpkg_path.name}: {e}")
        return {}