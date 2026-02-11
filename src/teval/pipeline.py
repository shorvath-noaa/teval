import logging
from pathlib import Path
import pandas as pd
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

def run_pipeline(config: TevalConfig):
    """
    Orchestrates the TEVAL evaluation workflow based on Pydantic configuration.
    """
    # Access via dot notation now!
    io = config.io
    data = config.data
    viz = config.viz

    # Ensure output dir exists
    io.output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    logger.info(f"Loading Ensemble Data from {io.input_dir}...")
    pattern = str(io.input_dir / io.ensemble_pattern)
    ds_ensemble = tio.load_ensemble(pattern)

    # Subset Time
    if data.time_slice:
        start, end = data.time_slice
        if isinstance(start, int):
            ds_ensemble = ds_ensemble.isel(time=slice(start, end))
        else:
            ds_ensemble = ds_ensemble.sel(time=slice(str(start), str(end)))
        logger.info(f"Time subset applied: {len(ds_ensemble.time)} steps.")

    # Load Hydrofabric
    logger.info("Loading Hydrofabric...")
    gdf_hydro = None
    
    # Logic to find the file
    hf_path = io.hydrofabric_path
    if not hf_path:
        # Auto-detect
        gpkg_files = list(io.input_dir.glob("*.gpkg"))
        if gpkg_files:
            hf_path = gpkg_files[0]
            
    if hf_path and hf_path.exists():
        gdf_hydro = tio.load_hydrofabric(hf_path)
        logger.info(f"Hydrofabric loaded: {hf_path.name}")
    else:
        logger.warning("No hydrofabric found. Spatial plots will be skipped.")

    # Load Observations
    obs_df = None
    if io.observations_file and io.observations_file.exists():
        logger.info(f"Loading local observations from {io.observations_file}...")
        obs_df = tobs.load_usgs_csv(io.observations_file)
        
    elif io.auto_download_usgs:
        logger.info("Local observations not found. Attempting auto-download from USGS...")
        try:
            # A. Get Date Range from Ensemble
            t_start = pd.to_datetime(ds_ensemble.time.min().values) - pd.Timedelta(days=1)
            t_end = pd.to_datetime(ds_ensemble.time.max().values) + pd.Timedelta(days=1)
            
            start_str = t_start.strftime("%Y-%m-%d")
            end_str = t_end.strftime("%Y-%m-%d")
            
            # B. Get Feature IDs
            gdf_gages = tio.load_hydrofabric(hf_path, 'flowpath-attributes')
            site_ids = gdf_gages.gage.dropna().unique().tolist() 
            
            logger.info(f"Querying USGS for {len(site_ids)} sites from {start_str} to {end_str}...")
            
            # C. Download
            df_new_obs = tobs.fetch_usgs_streamflow(site_ids, start_str, end_str)
            
            if not df_new_obs.empty:
                obs_df = df_new_obs
                logger.info(f"Downloaded USGS data.")
            else:
                logger.warning("USGS query returned no data.")
                
        except Exception as e:
            logger.error(f"Auto-download failed: {e}")

    if obs_df is not None:
        logger.info(f"Observations ready. Sites: {obs_df.shape[1]}")

    # 2. Calculate Statistics
    logger.info("Calculating Ensemble Statistics...")
    ds_stats = tstats.calculate_basics(ds_ensemble, 
                                       lower_quantile=config.stats.quantiles[0],
                                       upper_quantile=config.stats.quantiles[1])
    
    output_file = config.io.output_dir / "ensemble_stats.nc"
    logger.info(f"Saving NetCDF to {output_file}...")
    tio.save_ensemble_stats(ds_stats, output_file)
    
    # 3. Visualizations
    # A. Hydrographs
    if viz.hydrographs.enabled:
        logger.info("Generating Hydrographs...")
        hydro_dir = io.output_dir / "hydrographs"
        hydro_dir.mkdir(exist_ok=True)
        
        target_ids = viz.hydrographs.target_ids
        # If empty list, pick first 5
        if not target_ids:
            target_ids = ds_stats.feature_id.values[:5].tolist()
        
        for fid in target_ids:
            try:
                fig, ax = plt.subplots(figsize=(12, 6))
                
                series_obs = None
                gage_loc = gdf_gages.loc[fid].gage
                if obs_df is not None and gage_loc in obs_df.columns:
                    series_obs = obs_df[gage_loc]
                
                tviz.hydrograph(ds_stats, feature_id=fid, ax=ax, obs_series=series_obs)
                fig.savefig(hydro_dir / f"hydrograph_{fid}.png")
                plt.close(fig)
            except Exception as e:
                logger.error(f"Failed to plot hydrograph for {fid}: {e}")

    # B. Static Maps
    if viz.static_maps.enabled and gdf_hydro is not None:
        logger.info("Generating Static Maps...")
        map_dir = io.output_dir / "maps"
        map_dir.mkdir(exist_ok=True)
        
        for var in viz.static_maps.variables:
            try:
                fig, ax = plt.subplots(figsize=(10, 10))
                tviz.map_network(
                    gdf_hydro, ds_stats, 
                    var_name=var, 
                    ax=ax, 
                    add_basemap=viz.static_maps.basemap
                )
                fig.savefig(map_dir / f"map_{var}.png")
                plt.close(fig)
            except Exception as e:
                logger.error(f"Failed to plot map for {var}: {e}")

    # C. Interactive
    if viz.interactive_map.enabled and gdf_hydro is not None:
        logger.info("Generating Interactive Map...")
        out_html = io.output_dir / "interactive_map.html"
        try:
            tinteractive.map_folium(
                gdf_hydro, ds_stats,
                var_name=viz.interactive_map.variable,
                output_html=str(out_html)
            )
        except Exception as e:
             logger.error(f"Failed to generate interactive map: {e}")

    # D. Animation
    anim = viz.animation
    if anim.enabled and gdf_hydro is not None:
        logger.info("Generating Animation...")
        out_gif = io.output_dir / f"animation_{anim.variable}.gif"
        try:
            tanim.animate_network(
                gdf_hydro, ds_stats,
                output_path=str(out_gif),
                var_name=anim.variable,
                fps=anim.fps,
                log_scale=anim.log_scale,
                cmap_name=anim.cmap
            )
        except Exception as e:
            logger.error(f"Failed to generate animation: {e}")

    logger.info(f"Pipeline finished. Results in {io.output_dir}")