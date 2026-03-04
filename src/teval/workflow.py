
from teval.obs import usgs
import xarray as xr
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from typing import Dict, List
import logging
from zipfile import Path

from teval.config import IOConfig, MetricsConfig, StatsConfig, VizConfig
from teval.utils import find_tailwater_feature, parse_run_directory
from teval.metrics import deterministic as det
from teval.metrics import significance as sig
import teval.viz.static as tviz
import teval.viz.animation as tanim
import teval.viz.interactive as tinter


logger = logging.getLogger(__name__)

# Function for initializing domains based on directory structure and configuration options
# def initialize_domains(io: IOConfig,
#                        metrics: MetricsConfig,
#                        viz: VizConfig) -> List[Domain]:
#     """
#     TODO: UPDATE DOCSTRING
#     Scans an input directory, parses subfolder names, groups them by domain,
#     and returns a list of prepared Domain objects.
#     """ 
#     # Unpack configuration options
#     load_gpkgs = metrics.enabled or viz.static_maps.enabled or viz.interactive_map.enabled
#     fetch_obs = metrics.enabled or viz.hydrographs.enabled
    
#     ensemble_file = None
#     if io.ensemble_netcdf_dir:
#         ensemble_file = glob.glob(str(io.ensemble_netcdf_dir / "*ensemble.nc"))
    
#     if io.troute_netcdf_dir:
#         subdirs = [p for p in io.troute_netcdf_dir.iterdir() if p.is_dir() and p.name.endswith("_output")]
#         domain_map = {}
        
#         for folder in subdirs:
#             # Parse Folder Name
#             formulation_name, domain_name = parse_run_directory(folder)
            
#             if not formulation_name or not domain_name:
#                 continue
            
#             gpkg_path = None
#             if load_gpkgs:
#                 if domain_name=="CONUS":
#                     def make_case_insensitive(s):
#                         return "".join(f"[{c.lower()}{c.upper()}]" if c.isalpha() else c for c in s)

#                     pattern = f"*{make_case_insensitive(domain_name)}*.gpkg"
#                     gpkg_path = list(io.hydrofabric_dir.glob(pattern))
#                 else:
#                     gpkg_path = list(io.hydrofabric_dir.glob(f"*{domain_name}*.gpkg"))
            
#             gage_id = None
#             if fetch_obs:
#                 gage_id = domain_name
#                 if io.observations_file and io.observations_file.exists():
#                     obs_file_path = io.observations_file
            
#             # Set or Create Domain Key
#             if domain_name not in domain_map:
#                 domain_map[domain_name] = {
#                     "formulations": {},
#                     "hydrofabric": gpkg_path[0],
#                     "gage_obs": {'domain_name': [gage_id], 'obs_file': [obs_file_path]} if gage_id else [],
#                 }
            
#             # Find the NetCDF file inside
#             nc_files = list(folder.glob("*.nc"))
#             if not nc_files:
#                 logger.warning(f"No NetCDF file found in {folder.name}")
#                 continue
            
#             # Add Formulation to Domain
#             domain_map[domain_name]["formulations"][formulation_name] = nc_files[0]
    
#     domain_map[domain_name]["ensemble"] = ensemble_file if ensemble_file else []
    
#     return domain_map


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
# def load_domain_data(domain_dict: Dict[str, Dict]) -> List[Domain]:
#     """
#     Given a domain map with formulation paths, hydrofabric paths, and gage IDs, initializes Domain objects and loads their data.
#     """
#     function_map = {
#         'formulations': _process_formulation_files,
#         'hydrofabric': _process_hydrofabric,
#         'gage_obs': _fetch_observations
#         }
    
#     results = {}
#     for key, value in domain_dict.items():
#         if key in function_map:
#             if key=='formulations':
#                 results[key] = {'combined': [], 'ensemble_members': []}
#                 results[key]['combined'], results[key]['ensemble_members'], t_min, t_max = function_map[key](value) # Return start and end time for fetching observations
#             elif key=='gage_obs':
#                 results[key] = function_map[key](value, t_min=t_min, t_max=t_max)
#             else:
#                 results[key], all_gage_ids = function_map[key](value)
#                 domain_gage_str = domain_dict.get('gage_obs', {}).get('domain_name', [])
#                 if not domain_gage_str=="CONUS":
#                     domain_dict['gage_obs']['gage_list'] = list(set(domain_dict.get('gage_obs', {}).get('domain_name', []) + all_gage_ids)) # Update gage_obs list with any gage IDs found in hydrofabric
#                 else:
#                     domain_dict['gage_obs']['gage_list'] = all_gage_ids
#         else:
#             results[key] = f"No specific function for {key}" # Handle keys with no specific function
            
#     return results

# def _process_formulation_files(files: Dict[str, Path]) -> xr.Dataset:
#     """
#     Loads multiple NetCDF files matching a pattern, extracts the formulation ID from attributes or filenames, and concatenates them into a single xarray Dataset.
#     """
#     datasets = []
    
#     for f in files.values():
#         ds = xr.open_dataset(f, engine='netcdf4')
#         datasets.append(ds)
    
#     combined_ds = xr.concat(
#         datasets, 
#         dim=pd.Index(files.keys(), name="formulation")
#         )
    
    
    
#     ds_mean = combined_ds.mean(dim="formulation", keep_attrs=True)
#     ds_median = combined_ds.median(dim="formulation", keep_attrs=True)
    
#     # Rename variables
#     ds_mean = ds_mean.rename({v: f"{v}_mean" for v in ds_mean.data_vars})
#     ds_median = ds_median.rename({v: f"{v}_median" for v in ds_median.data_vars})
    
#     # Merge
#     ds_stats = xr.merge([ds_mean, ds_median])
    
#     # Calculate Max/Min/Quantiles
#     num_members = len(files)
#     if num_members < 10:
#         # Fast path using optimized bottleneck C-code
#         ds_lower = combined_ds.min(dim="formulation", keep_attrs=True)
#         ds_upper = combined_ds.max(dim="formulation", keep_attrs=True)
#     else:
#         # Slow path for large ensembles
#         ds_lower = combined_ds.quantile(0.05, dim="formulation", keep_attrs=True).drop_vars('quantile')
#         ds_upper = combined_ds.quantile(0.95, dim="formulation", keep_attrs=True).drop_vars('quantile')
    
#     # Rename them to expected p05 and p95 so downstream plotting doesn't break
#     ds_lower = ds_lower.rename({v: f"{v}_p05" for v in ds_lower.data_vars})
#     ds_upper = ds_upper.rename({v: f"{v}_p95" for v in ds_upper.data_vars})
    
#     ds_stats = xr.merge([ds_stats, ds_lower, ds_upper])
    
#     ds_stats.attrs = combined_ds.attrs
#     ds_stats.attrs['description'] = 'Ensemble Statistics'
    
#     t_min = pd.to_datetime(combined_ds.time.min().values)
#     t_max = pd.to_datetime(combined_ds.time.max().values)
    
#     return ds_stats, combined_ds, t_min, t_max

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
    
    # Load Pre-Computed Ensemble (if it exists)
    if ensemble_file and ensemble_file.exists():
        logger.info(f"Loading pre-computed ensemble from {ensemble_file.name}")
        ds_stats = xr.open_dataset(ensemble_file, engine="netcdf4")
        
        # Get time bounds from the ensemble
        if 'time' in ds_stats.coords:
            t_min = pd.to_datetime(ds_stats.time.min().values)
            t_max = pd.to_datetime(ds_stats.time.max().values)

    # Load Raw Formulation Files (if they exist)
    if raw_files:
        logger.info(f"Loading {len(raw_files)} raw formulation files...")
        datasets = [xr.open_dataset(f, engine="netcdf4") for f in raw_files.values()]
        combined_ds = xr.concat(datasets, dim=pd.Index(raw_files.keys(), name="formulation"))
        
        # Get time bounds from raw files if we didn't get them from the ensemble
        if t_min is None:
            t_min = pd.to_datetime(combined_ds.time.min().values)
            t_max = pd.to_datetime(combined_ds.time.max().values)

    # Calculate Stats ONLY if we don't have them yet
    if ds_stats is None and combined_ds is not None:
        logger.info("Calculating ensemble statistics from raw files...")
        ds_mean = combined_ds.mean(dim="formulation", keep_attrs=True)
        ds_median = combined_ds.median(dim="formulation", keep_attrs=True)
        
        ds_mean = ds_mean.rename({v: f"{v}_mean" for v in ds_mean.data_vars})
        ds_median = ds_median.rename({v: f"{v}_median" for v in ds_median.data_vars})
        ds_stats = xr.merge([ds_mean, ds_median])
        
        # Fast min/max for small ensembles instead of slow quantiles
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

# def _fetch_observations(obs_dict: List, t_min: pd.Timestamp, t_max: pd.Timestamp) -> pd.DataFrame:
#     """
#     Fetches USGS observations for a given gage ID.
#     """
#     import pdb; pdb.set_trace()
#     gage_id = obs_dict.get('gage_list', [None]) if obs_dict else None
#     obs_file = obs_dict.get('obs_file', [None])[0] if obs_dict else None
#     obs_df = pd.DataFrame()
    
#     if obs_file and obs_file.exists():
#         obs_df = pd.read_parquet(obs_file) if obs_file.suffix==".parquet" else pd.read_csv(obs_file)
#         obs_df.set_index('time', inplace=True)
#     elif len(gage_id) > 0 and gage_id[0] is not None:
#         obs_df = usgs.fetch_usgs_streamflow(
#             gage_id,
#             str(t_min.date()), 
#             str(t_max.date()),
#             to_cms=True,
#             to_utc=True
#             )
    
#         if not obs_df.empty:
#             # Resample to hourly to match model output
#             #TODO: Do we want to do resample the observations? Or resample the modeled data?
#             obs_df = obs_df.resample("1h").mean().interpolate()
        
#     return obs_df

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

# Functions for producing visualizations based on the loaded domain data and calculated metrics
def produce_domain_specific_visualizations(domain_data: Dict[str, Dict], viz: VizConfig, io: IOConfig, stats: StatsConfig):
    """
    Given the loaded domain data and visualization config, produces and saves visualizations.
    """
    ds_stats = domain_data.get('formulations', {}).get('combined', xr.Dataset())
    ds_ensemble = domain_data.get('formulations', {}).get('ensemble_members', xr.Dataset())
    gdf = domain_data.get('hydrofabric', gpd.GeoDataFrame())
    obs_df = domain_data.get('gage_obs', pd.DataFrame())
    
    # Hydrographs
    if viz.hydrographs.enabled:
        # TODO: Move this logic to earlier in __main__, create any output directories once if needed...
        hydro_dir = io.output_dir / "hydrographs"
        hydro_dir.mkdir(exist_ok=True)
        
        target_ids = viz.hydrographs.target_ids
        # If user didn't specify target_ids, try to find gaged locations, then fall back to tailwaters
        if not target_ids:
            valid_gage_df = gdf[gdf['gage'].isin(obs_df.columns.values)]
            valid_gage_df = valid_gage_df.loc[~valid_gage_df.index.duplicated(keep='first')]
            if not valid_gage_df.empty:
                target_ids = valid_gage_df.index.values
            else:
                target_ids = find_tailwater_feature(gdf.reset_index())
        
        for fid in target_ids:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            series_obs = None
            if valid_gage_df is not None and obs_df is not None:
                if fid in valid_gage_df.index:
                    gage_id = valid_gage_df.loc[fid].get('gage')
                    if gage_id and str(gage_id) in obs_df.columns:
                        series_obs = obs_df[str(gage_id)]
            
            tviz.hydrograph(
                ds_stats, 
                feature_id=fid, 
                ax=ax, 
                obs_series=series_obs,
                plot_uncertainty=viz.hydrographs.plot_uncertainty,
                plot_members=viz.hydrographs.plot_members,
                ensemble_ds=ds_ensemble,
                quantiles=stats.quantiles
            )
            fig.savefig(hydro_dir / f"hydrograph_{fid}.png")
            plt.close(fig)
    
    # Static Maps
    if viz.static_maps.enabled:
        if not gdf.empty:
            map_dir = io.output_dir / "maps"
            map_dir.mkdir(exist_ok=True)
            for var in viz.static_maps.variables:
                fig, ax = plt.subplots(figsize=(10, 10))
                tviz.map_network(gdf, ds_stats, var_name=var, ax=ax, add_basemap=viz.static_maps.basemap)
                fig.savefig(map_dir / f"map_{fid}_{var}.png")
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
    