from teval.obs import usgs
import xarray as xr
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from typing import Dict, List
import logging
from zipfile import Path
import multiprocessing
from joblib import Parallel, delayed
from dask.distributed import Lock
import gc

from teval.config import IOConfig, MetricsConfig, StatsConfig, VizConfig
from teval.utils import find_tailwater_feature, parse_run_directory
from teval.metrics import deterministic as det
from teval.metrics import significance as sig
import teval.viz.static as tviz
import teval.viz.animation as tanim
from teval.utils import Timer

logger = logging.getLogger(__name__)

def initialize_domains(io: IOConfig, metrics: MetricsConfig, viz: VizConfig) -> Dict:
    """
    Scans input directories, groups files by domain, and returns a prepared mapping.
    """
    load_gpkgs = metrics.enabled or viz.static_maps.enabled or viz.interactive_map.enabled
    fetch_obs = metrics.enabled or viz.hydrographs.enabled
    
    domain_map = {}
    
    # 1. First, discover raw T-Route outputs if provided
    if io.troute_netcdf_dir:
        subdirs = [p for p in io.troute_netcdf_dir.iterdir() if p.is_dir() and p.name.endswith("_output")]
        for folder in subdirs:
            formulation_name, domain_name = parse_run_directory(folder)
            if not formulation_name or not domain_name:
                continue
            
            # Initialize domain entry if new
            if domain_name not in domain_map:
                domain_map[domain_name] = _create_empty_domain_dict(domain_name, io, load_gpkgs, fetch_obs)
            
            # Find NetCDF file
            nc_files = list(folder.glob("*.nc"))
            if nc_files:
                domain_map[domain_name]["formulations"]["raw_files"][formulation_name] = nc_files[0]

    # 2. Second, discover pre-computed Ensembles if provided
    if io.ensemble_netcdf_dir:
        # Look for files like CONUS_ensemble.nc or just ensemble.nc inside a domain folder
        ensemble_files = list(io.ensemble_netcdf_dir.rglob("*.nc"))
        
        for e_file in ensemble_files:
            # Heuristic: Try to extract domain name from filename (e.g., "CONUS_ensemble.nc")
            # If your naming convention is different, adjust this extraction.
            domain_name = e_file.stem.split("_ensemble")[0] if "_ensemble" in e_file.stem else e_file.stem
            
            if domain_name not in domain_map:
                # If we only provided an ensemble dir (no raw troute dir), initialize the domain here
                domain_map[domain_name] = _create_empty_domain_dict(domain_name, io, load_gpkgs, fetch_obs)
            
            domain_map[domain_name]["formulations"]["ensemble_file"] = e_file

    if not domain_map:
        logger.error("No valid inputs found in troute_netcdf_dir or ensemble_netcdf_dir.")
        
    return domain_map

def _create_empty_domain_dict(domain_name: str, io: IOConfig, load_gpkgs: bool, fetch_obs: bool) -> dict:
    """Helper to initialize the dictionary structure for a new domain."""
    gpkg_path = None
    if load_gpkgs:
        if domain_name == "CONUS":
            def make_case_insensitive(s):
                return "".join(f"[{c.lower()}{c.upper()}]" if c.isalpha() else c for c in s)
            pattern = f"*{make_case_insensitive(domain_name)}*.gpkg"
            gpkgs = list(io.hydrofabric_dir.glob(pattern))
        else:
            gpkgs = list(io.hydrofabric_dir.glob(f"*{domain_name}*.gpkg"))
        gpkg_path = gpkgs[0] if gpkgs else None

    obs_info = {}
    if fetch_obs:
        obs_file_path = io.observations_file if (io.observations_file and io.observations_file.exists()) else None
        obs_info = {'domain_name': [domain_name], 'obs_file': [obs_file_path]}

    return {
        "formulations": {"raw_files": {}, "ensemble_file": None},
        "hydrofabric": gpkg_path,
        "gage_obs": obs_info,
    }

# Functions for loading domain data based on the domain map created in initialize_domains
def load_domain_data(domain_dict: Dict, io: IOConfig) -> Dict:
    results = {}
    
    # Process Formulations
    results['formulations'] = {'combined': None, 'ensemble_members': None}
    ds_stats, ds_members, t_min, t_max = _process_formulation_files(domain_dict['formulations'])
    
    results['formulations']['combined'] = ds_stats
    results['formulations']['ensemble_members'] = ds_members
    
    # Process Hydrofabric
    results['hydrofabric'], all_gage_ids = _process_hydrofabric(domain_dict['hydrofabric'])
    
    # Clean up the gage list (handling the CONUS logic)
    initial_gages = domain_dict.get('gage_obs', {}).get('domain_name', [])
    
    # Prevent the domain name itself from being queried as a USGS Gage
    if "CONUS" in initial_gages:
        initial_gages.remove("CONUS")
        
    # Combine any explicitly passed gages with the gages extracted from the hydrofabric
    gage_ids = list(set(initial_gages + all_gage_ids))
    
    # Fetch/Load Observations
    results['gage_obs'] = _fetch_observations(gage_ids, t_min, t_max, io)

    return results

def _process_formulation_files(formulation_dict: Dict) -> tuple:
    """
    Loads pre-computed ensemble if available, and raw members if available.
    Calculates stats only if a pre-computed ensemble is NOT provided.
    """
    raw_files = formulation_dict.get("raw_files", {})
    ensemble_file = formulation_dict.get("ensemble_file")
    
    ds_stats = None
    combined_ds = None
    t_min, t_max = None, None

    # 1. Load Pre-Computed Ensemble (if it exists)
    if ensemble_file and ensemble_file.exists():
        logger.info(f"Loading pre-computed ensemble from {ensemble_file.name}")
        # Use h5netcdf and chunk it directly
        ds_stats = xr.open_dataset(ensemble_file, engine="h5netcdf", chunks={'feature_id': 'auto'})
        
        if 'time' in ds_stats.coords:
            t_min = pd.to_datetime(ds_stats.time.min().values)
            t_max = pd.to_datetime(ds_stats.time.max().values)

    # 2. Load Raw Formulation Files (if they exist)
    if raw_files:
        logger.info(f"Loading {len(raw_files)} raw formulation files with Dask mfdataset...")
        
        read_lock = Lock("hdf5-read-lock")
        
        # open_mfdataset with parallel=True and h5netcdf for thread safety
        combined_ds = xr.open_mfdataset(
            paths=list(raw_files.values()),
            combine='nested',
            concat_dim="formulation", 
            engine="h5netcdf",
            chunks={},       # Match small disk chunks for streaming
            parallel=True,
            lock=read_lock  
        )
        
        # Assign coordinates properly
        combined_ds = combined_ds.assign_coords(formulation=list(raw_files.keys()))

        if t_min is None:
            t_min = pd.to_datetime(combined_ds.time.min().values)
            t_max = pd.to_datetime(combined_ds.time.max().values)

    # 3. Calculate Stats Lazily
    if ds_stats is None and combined_ds is not None:
        logger.info("Setting up lazy ensemble statistics calculations...")
        
        ds_mean = combined_ds.mean(dim="formulation", keep_attrs=True)
        ds_median = combined_ds.median(dim="formulation", keep_attrs=True)
        
        ds_mean = ds_mean.rename({v: f"{v}_mean" for v in ds_mean.data_vars})
        ds_median = ds_median.rename({v: f"{v}_median" for v in ds_median.data_vars})
        ds_stats = xr.merge([ds_mean, ds_median])
        
        if len(raw_files) < 10:
            ds_lower = combined_ds.min(dim="formulation", keep_attrs=True)
            ds_upper = combined_ds.max(dim="formulation", keep_attrs=True)
        else:
            ds_lower = combined_ds.quantile(0.05, dim="formulation", keep_attrs=True).drop_vars('quantile')
            ds_upper = combined_ds.quantile(0.95, dim="formulation", keep_attrs=True).drop_vars('quantile')

        ds_lower = ds_lower.rename({v: f"{v}_p05" for v in ds_lower.data_vars})
        ds_upper = ds_upper.rename({v: f"{v}_p95" for v in ds_upper.data_vars})
        
        ds_stats = xr.merge([ds_stats, ds_lower, ds_upper])
        ds_stats.attrs = combined_ds.attrs
        ds_stats.attrs['description'] = 'Ensemble Statistics'
        
        # NOTE: We purposefully DO NOT call ds_stats = ds_stats.compute() here!
        # It remains completely lazy until it streams to the NetCDF file in __main__.py
        
    elif ds_stats is None and combined_ds is None:
        raise ValueError("No ensemble file or raw formulation files found to process.")
        
    return ds_stats, combined_ds, t_min, t_max

def _process_hydrofabric(gpkg_path: str) -> gpd.GeoDataFrame:
    """
    Loads the hydrofabric GeoDataFrame and prepares the index for joining with model output.
    """
    gdf = gpd.GeoDataFrame()
    if gpkg_path:
        gdf = pd.merge(
            gpd.read_file(gpkg_path, layer='flowpaths')[['id','toid','hydroseq','order','geometry']],
            gpd.read_file(gpkg_path, layer='flowpath-attributes')[['id','gage']]
            )
        
        gdf['id'] = gdf['id'].str.replace(r'\D+', '', regex=True).astype(int)
        gdf['toid'] = gdf['toid'].str.replace(r'\D+', '', regex=True).astype(int)
        
        gdf.set_index('id', inplace=True)
        
        # Return list of gageIDs in the domain
        gage_ids = gdf['gage'].dropna().unique().tolist()
        
        # Re-project the gdf to lat/lon:
        gdf = gdf.to_crs(epsg=4326)
    
    return gdf, gage_ids

def _fetch_observations(gage_ids: List, t_min: pd.Timestamp, t_max: pd.Timestamp, io: IOConfig) -> pd.DataFrame:
    """Loads observations from file (Parquet/CSV) if provided, else queries USGS API."""
    obs_df = pd.DataFrame()
    
    if not gage_ids:
        return obs_df
        
    # 1. Load from file (if provided)
    if io.observations_file and io.observations_file.exists():
        file_path = io.observations_file
        logger.info(f"Loading observations from {file_path.suffix} file: {file_path}")
        
        if file_path.suffix == '.parquet':
            obs_df = pd.read_parquet(file_path)
        elif file_path.suffix == '.csv':
            obs_df = pd.read_csv(file_path)
        else:
            logger.error(f"Unsupported observation file format: {file_path.suffix}")
            return obs_df
            
        # Set index
        if 'time' in obs_df.columns:
            obs_df.set_index('time', inplace=True)
        elif 'datetime' in obs_df.columns:
            obs_df.set_index('datetime', inplace=True)
            
        obs_df.index = pd.to_datetime(obs_df.index)
        
        # Filter dataframe to only contain the requested gages
        # Convert all columns and requested gages to string to ensure safe matching
        valid_gages = list(set(str(g) for g in obs_df.columns) & set(str(g) for g in gage_ids))
        obs_df = obs_df[valid_gages]
        
    # 2. Fallback to API download
    elif io.auto_download_usgs:
        # Filter out any lingering non-numeric strings just in case
        clean_gages = [str(g) for g in gage_ids if str(g).isdigit()]
        
        if clean_gages:
            logger.info("Fetching USGS data via API...")
            obs_df = usgs.fetch_usgs_streamflow(
                clean_gages, 
                str(t_min.date()), 
                str(t_max.date()), 
                to_cms=True, 
                to_utc=True
            )
            
            # ONLY resample if fetched from API, matching old code logic
            if not obs_df.empty:
                obs_df = obs_df.resample("1h").mean().interpolate()
                
                if io.save_downloaded_obs:
                    obs_df.to_csv(io.save_downloaded_obs)
                
    else:
        logger.warning("No observation file provided and auto_download is disabled.")
        
    return obs_df

def _calc_row(sim_series: pd.Series, obs_series: pd.Series, source_name: str, 
              fid: int, gage_id_str: str, lat: float, lon: float, 
              metrics_config: MetricsConfig) -> dict:
    """Helper to calculate metrics for a single simulation series vs observations."""
    if sim_series.index.tz is not None:
        sim_series = sim_series.tz_localize(None)
        
    df_aligned = pd.concat([obs_series, sim_series], axis=1, join="inner").dropna()
    if len(df_aligned) < 5: return None
    
    obs_aligned = df_aligned.iloc[:, 0]
    sim_aligned = df_aligned.iloc[:, 1]
    
    row = {'feature_id': int(fid), 'gage_id': gage_id_str, 'lat': lat, 'lon': lon, 'source': source_name}
    
    for metric in metrics_config.variables:
        row[metric] = det.calculate_deterministic_metric(obs_aligned, sim_aligned, metric_name=metric)
    
    if metrics_config.bootstrap_enabled and 'nse' in metrics_config.variables:
        obs_vec, sim_vec = obs_aligned.values, sim_aligned.values
        res = sig.bootstrap_confidence_interval(
            obs_vec, sim_vec, det.nse, 
            n_samples=metrics_config.bootstrap_samples, 
            ci_level=metrics_config.confidence_level
        )
        row['sig_class'] = res['significance']
        
    return row
            
# Functions for calculating metrics based on the loaded domain data
def calculate_metrics(domain_data: Dict[str, Dict], 
                      metrics: MetricsConfig,
                      sim_var: str = 'streamflow_mean'
                      ) -> Dict[str, Dict]:
    """
    Given the loaded domain data, calculates specified metrics and returns results in a structured format.
    """
    metric_results = []
    
    obs_df = domain_data.get('gage_obs', pd.DataFrame())
    gdf = domain_data.get('hydrofabric', gpd.GeoDataFrame())
    ds_stats = domain_data.get('formulations', {}).get('combined', xr.Dataset())
    ds_ensemble = domain_data.get('formulations', {}).get('ensemble_members', xr.Dataset())
    
    if not obs_df.empty and not gdf.empty and len(ds_stats.dims) > 0:
        # Map FeatureID -> GageID if possible
        fid_to_gage = {}
        if  'gage' in gdf.columns:
            # Create mapping from index (feature_id) to gage column
            valid_gage_df = gdf[~gdf['gage'].isnull()]
            if not valid_gage_df.empty:
                fid_to_gage = dict(zip(valid_gage_df.index, valid_gage_df['gage']))
                valid_fids = [f for f, g in fid_to_gage.items() if str(g) in obs_df.columns]
                valid_gage_df = valid_gage_df.loc[valid_fids]
                valid_gage_df = valid_gage_df.loc[~valid_gage_df.index.duplicated(keep='first')]
            else:
                logger.warning("No valid gage entries found in hydrofabric for mapping.")
        
        logger.info("Converting Xarray slice to Pandas for fast metric calculation...")
        sim_df_mean = ds_stats[sim_var].sel(feature_id=valid_fids).to_pandas()
        
        # If calculating per formulation, convert the full ensemble to pandas too
        sim_df_members = None
        formulation_names = []
        
        if metrics.per_formulation and ds_ensemble is not None:
            # Convert to DataFrame: MultiIndex (time, formulation) -> columns are feature_ids
            # Or just loop formulation dimension and make a dict of dataframes
            formulation_names = ds_ensemble.formulation.values
            sim_df_members = {}
            for form in formulation_names:
                sim_df_members[form] = ds_ensemble['streamflow'].sel(formulation=form, feature_id=valid_fids).to_pandas()
        
        for fid in valid_fids:
            gage_id_str = str(fid_to_gage[fid])
            obs_series = obs_df[gage_id_str].tz_localize(None) if obs_df[gage_id_str].index.tz else obs_df[gage_id_str]
            
            geom = valid_gage_df.loc[fid].geometry.centroid if fid in valid_gage_df.index else None
            lat, lon = (geom.y, geom.x) if geom else (None, None)

            # Calculate Ensemble Mean
            row_mean = _calc_row(
                sim_series=sim_df_mean[fid], 
                obs_series=obs_series, 
                source_name="ensemble_mean", 
                fid=fid, 
                gage_id_str=gage_id_str, 
                lat=lat, 
                lon=lon, 
                metrics_config=metrics
            )
            if row_mean: metric_results.append(row_mean)
            
            # Calculate Per Formulation
            if metrics.per_formulation and sim_df_members:
                for form in formulation_names:
                    row_form = _calc_row(
                        sim_series=sim_df_members[form][fid], 
                        obs_series=obs_series, 
                        source_name=form, 
                        fid=fid, 
                        gage_id_str=gage_id_str, 
                        lat=lat, 
                        lon=lon, 
                        metrics_config=metrics
                    )
                    if row_form: metric_results.append(row_form)
    
    return metric_results

def _render_single_hydrograph(fid, ds_stats, obs_df, valid_gage_df, viz, stats, ds_ensemble, hydro_dir, metrics_df):
    """Joblib worker function to render a single hydrograph safely with metrics."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    series_obs = None
    if valid_gage_df is not None and obs_df is not None:
        if fid in valid_gage_df.index:
            gage_id = valid_gage_df.loc[fid].get('gage')
            if gage_id and str(gage_id) in obs_df.columns:
                series_obs = obs_df[str(gage_id)]
                
    # Isolate metrics for just this specific Feature ID
    fid_metrics = None
    if metrics_df is not None and not metrics_df.empty:
        fid_metrics = metrics_df[metrics_df['feature_id'] == fid]
    
    tviz.hydrograph(
        ds_stats, 
        feature_id=fid, 
        ax=ax, 
        obs_series=series_obs,
        plot_uncertainty=viz.hydrographs.plot_uncertainty,
        plot_members=viz.hydrographs.plot_members,
        ensemble_ds=ds_ensemble,
        quantiles=stats.quantiles,
        metrics_df=fid_metrics
    )
    fig.savefig(hydro_dir / f"hydrograph_{fid}.png", bbox_inches='tight')
    plt.close(fig)
    gc.collect() 


# Functions for producing visualizations based on the loaded domain data and calculated metrics
def produce_domain_specific_visualizations(domain_data: Dict, viz: VizConfig, io: IOConfig, stats: StatsConfig):
    """
    Given the loaded domain data and visualization config, produces and saves visualizations.
    """
    ds_stats = domain_data.get('formulations', {}).get('combined', xr.Dataset())
    ds_ensemble = domain_data.get('formulations', {}).get('ensemble_members', xr.Dataset())
    gdf = domain_data.get('hydrofabric', gpd.GeoDataFrame())
    obs_df = domain_data.get('gage_obs', pd.DataFrame())
    
    # Extract calculated metrics to pass to the plots
    metrics_list = domain_data.get('metrics', [])
    metrics_df = pd.DataFrame(metrics_list) if metrics_list else None
    
    # 1. Hydrographs
    if viz.hydrographs.enabled:
        hydro_dir = io.output_dir / "hydrographs"
        hydro_dir.mkdir(exist_ok=True)
        
        target_ids = viz.hydrographs.target_ids
        valid_gage_df = None
        if not target_ids:
            valid_gage_df = gdf[gdf['gage'].isin(obs_df.columns.values)]
            valid_gage_df = valid_gage_df.loc[~valid_gage_df.index.duplicated(keep='first')]
            if not valid_gage_df.empty:
                target_ids = valid_gage_df.index.values
            else:
                target_ids = find_tailwater_feature(gdf.reset_index())
        
        valid_targets = np.intersect1d(target_ids, ds_stats.feature_id.values)
        logger.info(f"Pre-loading data for {len(valid_targets)} hydrographs into RAM...")
        
        ds_stats_subset = ds_stats.sel(feature_id=valid_targets).compute()
        
        ds_ensemble_subset = None
        if viz.hydrographs.plot_members and ds_ensemble is not None:
            if 'feature_id' in ds_ensemble.dims:
                ds_ensemble_subset = ds_ensemble.sel(feature_id=valid_targets).compute()

        logger.info(f"Generating {len(valid_targets)} hydrographs in parallel...")
        
        with Timer("Plotting Hydrographs"):
            n_cores = max(1, multiprocessing.cpu_count() - 1)
            Parallel(n_jobs=n_cores)(
                delayed(_render_single_hydrograph)(fid, ds_stats_subset, obs_df, valid_gage_df, viz, stats, ds_ensemble_subset, hydro_dir, metrics_df) 
                for fid in valid_targets
            )
    
    # Static Maps
    if viz.static_maps.enabled:
        if not gdf.empty:
            map_dir = io.output_dir / "maps"
            map_dir.mkdir(exist_ok=True)
            for var in viz.static_maps.variables:
                fig, ax = plt.subplots(figsize=(10, 10))
                tviz.map_network(gdf, ds_stats, var_name=var, ax=ax, add_basemap=viz.static_maps.basemap)
                # THE FIX: Name the map using 'domain', not a leftover 'fid' from a previous loop
                fig.savefig(map_dir / f"map_domain_{var}.png", bbox_inches='tight')
                plt.close(fig)
    
    # Animation
    if viz.animation.enabled:
        logger.info("Generating Animation...")
        anim_dir = io.output_dir / "animations"
        anim_dir.mkdir(exist_ok=True)
        
        # 1. Filter by stream order
        if 'order' in gdf.columns:
            gdf_anim = gdf[gdf['order'] >= viz.animation.min_stream_order]
        else:
            logger.warning("Stream order not found in geopackage, using all paths.")
            gdf_anim = gdf
            
        # 2. Extract valid feature IDs and drop duplicates
        anim_fids = gdf_anim.index.values
        common_anim_ids = np.intersect1d(anim_fids, ds_stats.feature_id.values)
        
        gdf_anim = gdf_anim.loc[common_anim_ids]
        gdf_anim = gdf_anim[~gdf_anim.index.duplicated(keep='first')] # Ensure unique IDs
        
        # 3. Time subsetting (e.g., '1H', '1D', '3D', '1W')
        time_step_str = str(viz.animation.time_step).strip().upper()
        
        try:
            # If the user passed a raw integer (e.g., "168")
            step = int(time_step_str)
        except ValueError:
            # If the user passed a string, calculate the dataset's native time resolution
            if len(ds_stats.time) >= 2:
                # Get difference between first two time steps
                dt_native = pd.Timedelta(ds_stats.time.values[1] - ds_stats.time.values[0])
                
                # Handle 'W' (weeks) manually since pd.to_timedelta deprecated 'W' in recent versions
                if time_step_str.endswith('W'):
                    val = int(time_step_str[:-1]) if len(time_step_str) > 1 else 1
                    dt_target = pd.Timedelta(days=val * 7)
                else:
                    # pd.to_timedelta handles '1H', '1D', '3D', etc. perfectly
                    dt_target = pd.to_timedelta(time_step_str)
                    
                # Calculate how many integer steps to jump per frame
                step = max(1, int(dt_target / dt_native))
            else:
                step = 1
                
        ds_anim_sliced = ds_stats.isel(time=slice(0, None, step))
        ds_anim_sliced = ds_anim_sliced.sel(feature_id=common_anim_ids).sortby('feature_id')
        
        # 4. Generate
        domain_name = domain_data.get('gage_obs', {}).get('domain_name', ['domain'])[0]
        out_gif = anim_dir / f"streamflow_animation_{domain_name}.gif"
        
        with Timer("Generating GIF Animation"):
            tanim.animate_network(
                gdf=gdf_anim,
                stats_ds=ds_anim_sliced,
                output_path=str(out_gif),
                var_name=viz.animation.variable,
                fps=viz.animation.fps,
                add_basemap=True
            )

def plot_metrics_on_map(metrics_df: pd.DataFrame, variable: str, output_path: Path, title: str = None):
    """
    Plots a specified metric on a map using the provided metrics DataFrame.
    Acts as a wrapper for the static visualization module to handle dynamic 
    titles and file paths per formulation.
    """
    if metrics_df.empty:
        logger.warning(f"Metrics dataframe is empty. Skipping map for {title or variable}.")
        return

    tviz.map_metrics(
        metrics_df=metrics_df,
        variable=variable,
        output_path=output_path,
        title=title
    )