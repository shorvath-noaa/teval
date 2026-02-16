import logging
from pathlib import Path
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import teval.obs.usgs as tobs
import teval.io as tio
import teval.stats as tstats
import teval.viz.static as tviz
import teval.viz.interactive as tinteractive
import teval.viz.animation as tanim
from teval.config import TevalConfig

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_time_range(ds_ensemble, ds_stats):
    ds = ds_ensemble if ds_ensemble is not None else ds_stats
    if ds is None:
        raise ValueError("Cannot determine time range: neither Ensemble nor Stats data is loaded.")
    t_start = pd.to_datetime(ds.time.min().values) - pd.Timedelta(days=1)
    t_end = pd.to_datetime(ds.time.max().values) + pd.Timedelta(days=1)
    return t_start.strftime("%Y-%m-%d"), t_end.strftime("%Y-%m-%d")

def load_observations(io_config, start_date, end_date):
    obs_df = None
    if io_config.observations_file and io_config.observations_file.exists():
        logger.info(f"Loading local observations from {io_config.observations_file}...")
        try:
            obs_df = tobs.load_usgs_csv(io_config.observations_file)
        except Exception as e:
            logger.error(f"Failed to read local obs file: {e}")
    elif io_config.auto_download_usgs:
        logger.info("Local observations not found. Attempting auto-download from USGS...")
        if not io_config.hydrofabric_path or not io_config.hydrofabric_path.exists():
            logger.warning("Cannot auto-download USGS data: Hydrofabric (for Gage IDs) is missing.")
            return None
        try:
            logger.info("Reading Gage IDs from Hydrofabric flowpath-attributes...")
            gdf_gages = tio.load_hydrofabric(io_config.hydrofabric_path, layer='flowpath-attributes')
            if 'gage' in gdf_gages.columns:
                site_ids = gdf_gages['gage'].dropna().unique().astype(str).tolist()
                site_ids = [s for s in site_ids if len(s) >= 8] 
            else:
                logger.warning("Hydrofabric 'flowpath-attributes' layer missing 'gage' column.")
                return None
            if not site_ids:
                logger.warning("No linked Gage IDs found in hydrofabric.")
                return None
            logger.info(f"Querying USGS for {len(site_ids)} sites from {start_date} to {end_date}...")
            df_new = tobs.fetch_usgs_streamflow(site_ids, start_date, end_date)
            if not df_new.empty:
                obs_df = df_new
                if io_config.save_downloaded_obs is not None:
                    save_path = io_config.save_downloaded_obs
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    obs_df.to_csv(save_path)
                    logger.info(f"Downloaded USGS data saved to {save_path}")
                else:
                    logger.info("Downloaded data loaded (not saving to disk).")
            else:
                logger.warning("USGS query returned no data.")
        except Exception as e:
            logger.error(f"Auto-download failed: {e}")
    return obs_df

def run_pipeline(config: TevalConfig):
    io = config.io
    data = config.data
    stats_cfg = config.stats
    viz = config.viz

    io.output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Ensemble
    ds_ensemble = None
    should_load_ensemble = stats_cfg.enabled or viz.hydrographs.plot_members
    
    if should_load_ensemble:
        logger.info(f"Loading Ensemble Data from {io.input_dir}...")
        try:
            pattern = str(io.input_dir / io.ensemble_pattern)
            ds_ensemble = tio.load_ensemble(pattern)
            if data.time_slice:
                start, end = data.time_slice
                if isinstance(start, int):
                    ds_ensemble = ds_ensemble.isel(time=slice(start, end))
                else:
                    ds_ensemble = ds_ensemble.sel(time=slice(str(start), str(end)))
                logger.info(f"Time subset applied: {len(ds_ensemble.time)} steps.")
        except Exception as e:
            logger.error(f"Failed to load ensemble data: {e}")
            if stats_cfg.enabled: raise 
    else:
        logger.info("Skipping ensemble load (Stats disabled and not plotting members).")

    # 2. Statistics
    ds_stats = None
    stats_path = io.output_dir / io.stats_file

    if stats_cfg.enabled:
        if ds_ensemble is None:
             raise ValueError("Stats calculation enabled but ensemble data not loaded.")
        logger.info("Calculating Ensemble Statistics...")
        ds_stats = tstats.calculate_basics(ds_ensemble)
        logger.info(f"Saving statistics to {stats_path}...")
        ds_stats.to_netcdf(stats_path)
    else:
        if stats_path.exists():
            logger.info(f"Loading pre-calculated statistics from {stats_path}...")
            ds_stats = xr.open_dataset(stats_path)
            if data.time_slice:
                try:
                    start, end = data.time_slice
                    if isinstance(start, int):
                        ds_stats = ds_stats.isel(time=slice(start, end))
                    else:
                        ds_stats = ds_stats.sel(time=slice(str(start), str(end)))
                    logger.info(f"Time subset applied to stats: {len(ds_stats.time)} steps.")
                except Exception as e:
                    logger.warning(f"Could not apply time_slice to loaded stats: {e}")
        else:
            msg = f"Stats calculation disabled, but {stats_path} does not exist."
            logger.error(msg)
            raise FileNotFoundError(msg)

    # 3. Hydrofabric
    hf_path = io.hydrofabric_path
    if not hf_path:
        gpkg_files = list(io.input_dir.glob("*.gpkg"))
        if gpkg_files: 
            hf_path = gpkg_files[0]
            io.hydrofabric_path = hf_path 

    if not hf_path or not hf_path.exists():
        logger.warning("No hydrofabric found. Spatial plots and gage lookups will be skipped.")

    # 4. Observations
    obs_df = None
    gdf_gages = None 
    try:
        start_date, end_date = get_time_range(ds_ensemble, ds_stats)
        obs_df = load_observations(io, start_date, end_date)
        if hf_path and hf_path.exists() and (viz.hydrographs.enabled and obs_df is not None):
             try:
                gdf_gages = tio.load_hydrofabric(hf_path, layer='flowpath-attributes')
             except Exception:
                pass
        if obs_df is not None:
             logger.info(f"Observations ready. Sites: {obs_df.shape[1]}")
    except Exception as e:
        logger.error(f"Observation setup failed: {e}")

    # 5. Visualizations
    gdf_hydro = None
    def ensure_hydrofabric_loaded():
        nonlocal gdf_hydro
        if gdf_hydro is None and hf_path and hf_path.exists():
            logger.info(f"Loading Hydrofabric Geometries from {hf_path.name}...")
            gdf_hydro = tio.load_hydrofabric(hf_path)
        return gdf_hydro

    # A. Hydrographs
    if viz.hydrographs.enabled:
        logger.info("Generating Hydrographs...")
        hydro_dir = io.output_dir / "hydrographs"
        hydro_dir.mkdir(exist_ok=True)
        
        target_ids = viz.hydrographs.target_ids
        if not target_ids:
            target_ids = ds_stats.feature_id.values[:5].tolist()
            
        for fid in target_ids:
            try:
                fig, ax = plt.subplots(figsize=(12, 6))
                
                series_obs = None
                if gdf_gages is not None and obs_df is not None:
                    if fid in gdf_gages.index:
                        gage_id = gdf_gages.loc[fid].get('gage')
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
                    quantiles=stats_cfg.quantiles
                )
                fig.savefig(hydro_dir / f"hydrograph_{fid}.png")
                plt.close(fig)
            except Exception as e:
                logger.error(f"Failed to plot hydrograph for {fid}: {e}")

    # B. Static Maps
    if viz.static_maps.enabled:
        gdf = ensure_hydrofabric_loaded()
        if gdf is not None:
            logger.info("Generating Static Maps...")
            map_dir = io.output_dir / "maps"
            map_dir.mkdir(exist_ok=True)
            for var in viz.static_maps.variables:
                try:
                    fig, ax = plt.subplots(figsize=(10, 10))
                    tviz.map_network(gdf, ds_stats, var_name=var, ax=ax, add_basemap=viz.static_maps.basemap)
                    fig.savefig(map_dir / f"map_{var}.png")
                    plt.close(fig)
                except Exception as e:
                    logger.error(f"Failed to plot map for {var}: {e}")

    # C. Interactive
    if viz.interactive_map.enabled:
        gdf = ensure_hydrofabric_loaded()
        if gdf is not None:
            logger.info("Generating Interactive Map...")
            out_html = io.output_dir / "interactive_map.html"
            try:
                tinteractive.map_folium(gdf, ds_stats, var_name=viz.interactive_map.variable, output_html=str(out_html))
            except Exception as e:
                 logger.error(f"Failed to generate interactive map: {e}")

    # D. Animation
    anim = viz.animation
    if anim.enabled:
        gdf = ensure_hydrofabric_loaded()
        if gdf is not None:
            logger.info("Generating Animation...")
            out_gif = io.output_dir / f"animation_{anim.variable}.gif"
            try:
                tanim.animate_network(gdf, ds_stats, output_path=str(out_gif), var_name=anim.variable, fps=anim.fps, log_scale=anim.log_scale, cmap_name=anim.cmap)
            except Exception as e:
                logger.error(f"Failed to generate animation: {e}")

    logger.info(f"Pipeline finished. Results in {io.output_dir}")