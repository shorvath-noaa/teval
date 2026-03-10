import argparse
import sys
import time
import logging
import pandas as pd
import xarray as xr
from teval.config import TevalConfig, generate_default_config, generate_config_help
from teval import workflow
from teval.utils import Timer
import os
import multiprocessing
from dask.distributed import Client

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Run the TEVAL Evaluation Pipeline")
    parser.add_argument(
        "-c", "--config", 
        type=str, 
        default="teval_default_config.yaml",
        help="Path to the configuration YAML file."
    )
    parser.add_argument(
        "--init", 
        action="store_true",
        help="Generate a default 'teval_config.yaml' file in the current directory."
    )
    parser.add_argument(
        "--help-config", 
        action="store_true",
        help="Print a detailed guide of all configuration parameters."
    )
    
    args = parser.parse_args()
    
    if args.init:
        generate_default_config("teval_default_config.yaml")
        print("generated default configuration: teval_default_config.yaml")
        sys.exit(0)
    
    if args.help_config:
        print(generate_config_help())
        sys.exit(0)

    # Load and Validate Configuration
    config = TevalConfig.from_yaml(args.config)
    
    # DASK INITIALIZATION (Thread-safe with h5netcdf)
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

    slurm_cores = os.environ.get("SLURM_CPUS_PER_TASK")
    num_cores = int(slurm_cores) if slurm_cores else max(1, multiprocessing.cpu_count() - 1)
    
    client = Client(processes=False, n_workers=1, threads_per_worker=num_cores)
    
    logger.info(f"Initialized Dask Client with 1 worker and {num_cores} threads.")
    logger.info(f"Dask Dashboard: {client.dashboard_link}")
    

    # Create domain mapping containing individual formulations, gpkg paths, USGS gage information, etc.
    logger.info("Starting Domain Initialization...")
    run_domain_processing = any([
        config.metrics.enabled,
        config.viz.hydrographs.enabled,
        config.viz.static_maps.enabled,
        config.viz.animation.enabled
    ])
    
    metrics_df = pd.DataFrame()
    out_csv = config.io.output_dir / config.metrics.metrics_output_file if config.metrics.metrics_output_file else None

    if run_domain_processing:
        logger.info("Starting Domain Initialization...")
        domain_map = workflow.initialize_domains(config.io, config.metrics, config.viz)
        
        logger.info("Starting Domain Processing...")
        domain_data = {}
        metrics_list = []
        
        for key, value in domain_map.items():
            with Timer(f"1. Loading Raw Data ({key})"):
                domain_data[key] = workflow.load_domain_data(value, config.io)
                
                if config.data.time_slice and len(config.data.time_slice) == 2:
                    t_start, t_end = config.data.time_slice
                    logger.info(f"Slicing domain data time from {t_start} to {t_end}...")
                    
                    if domain_data[key]['formulations']['combined'] is not None:
                        domain_data[key]['formulations']['combined'] = domain_data[key]['formulations']['combined'].sel(time=slice(t_start, t_end))
                        
                    if domain_data[key]['formulations']['ensemble_members'] is not None:
                        domain_data[key]['formulations']['ensemble_members'] = domain_data[key]['formulations']['ensemble_members'].sel(time=slice(t_start, t_end))
            
            with Timer(f"2. Streaming NetCDF to Disk ({key})"):
                if value['formulations'].get('ensemble_file') is None:
                    out_nc = config.io.output_dir / f"{key}_ensemble.nc"
                    logger.info(f"Streaming computed ensemble directly to {out_nc}...")
                    ds_to_save = domain_data[key]['formulations']['combined'].astype('float32')
                    ds_to_save.to_netcdf(out_nc, engine="h5netcdf")
                    domain_data[key]['formulations']['combined'] = xr.open_dataset(
                        out_nc, engine="h5netcdf", chunks={'time': -1, 'feature_id': 'auto'}
                    )
                else:
                    logger.info(f"Skipping ensemble save; using pre-computed file: {value['formulations']['ensemble_file']}")
                    domain_data[key]['formulations']['combined'] = domain_data[key]['formulations']['combined'].chunk(
                        {'time': -1, 'feature_id': 'auto'}
                    )
            
            # Check config before calculating metrics
            if config.metrics.enabled:
                with Timer(f"3. Calculating Metrics ({key})"):
                    domain_data[key]['metrics'] = workflow.calculate_metrics(domain_data[key], config.metrics)
                    metrics_list.append(pd.DataFrame(domain_data[key]['metrics']))
            
            # Check config before plotting heavy visualizations
            viz_enabled = any([config.viz.hydrographs.enabled, config.viz.static_maps.enabled, config.viz.animation.enabled])
            if viz_enabled:
                with Timer(f"4. Generating Visualizations ({key})"):
                    workflow.produce_domain_specific_visualizations(domain_data[key], config.viz, config.io, config.stats)
            
        if metrics_list:
            metrics_df = pd.concat(metrics_list)
            if not metrics_df.empty and out_csv:
                metrics_df.to_csv(out_csv, index=False)
                logger.info(f"Metrics saved to {out_csv}")
    else:
        logger.info("Domain processing disabled based on configuration. Skipping heavy I/O.")
        
        if config.viz.metrics_maps.enabled or config.viz.interactive_map.enabled:
            if out_csv and out_csv.exists():
                logger.info(f"Loading pre-calculated metrics directly from {out_csv}")
                metrics_df = pd.read_csv(out_csv)
            else:
                logger.error(f"Cannot find metrics file {out_csv} to generate maps! Please run with metrics: enabled: true first.")

    # Post-processing maps and html
    if not metrics_df.empty:
        # Static Metric Maps
        if config.viz.metrics_maps.enabled:
            sources = metrics_df['source'].unique()
            for metric in config.viz.metrics_maps.variables:
                for src in sources:
                    src_df = metrics_df[metrics_df['source'] == src]
                    if metric in src_df.columns:
                        workflow.plot_metrics_on_map(
                            metrics_df=src_df,
                            variable=metric, 
                            output_path=config.io.output_dir / f"map_{metric}_{src}.png",
                            title=f"{metric.upper()} ({src})"
                        )
                    else:
                        logger.warning(f"Metric '{metric}' not found in dataframe for source '{src}'. Skipping map.")

        # Interactive Map
        if config.viz.interactive_map.enabled:
            from teval.viz.interactive import plot_interactive_metrics_map
            map_out = config.io.output_dir / "interactive_metrics_map.html"
            plot_interactive_metrics_map(metrics_df, output_path=map_out)

if __name__ == "__main__":
    main()