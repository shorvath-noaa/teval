import argparse
import sys
import logging
import pandas as pd
from teval.config import TevalConfig, generate_default_config, generate_config_help
from teval.config import TevalConfig
from teval import workflow
from teval.viz.points import plot_domain_metrics
# from teval.framework.evaluator import Evaluator

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
    
    # Create domain mapping containing individual formulations, gpkg paths, USGS gage information, etc.
    logger.info("Starting Domain Initialization...")
    domain_map = workflow.initialize_domains(config.io, config.metrics, config.viz)
    
    
    # TODO: Add parallel option
    logger.info("Starting Domain Processing...")
    domain_data = {}
    metrics_list = []
    
    for key, value in domain_map.items():
        # Read files ngen files, hydrofabric, and observations for each domain (if applicable)
        domain_data[key] = workflow.load_domain_data(value, config.io)
        # Only save the ensemble NetCDF if we generated it from scratch this run
        if value['formulations'].get('ensemble_file') is None:
            out_nc = config.io.output_dir / f"{key}_ensemble.nc"
            logger.info(f"Saving newly computed ensemble to {out_nc}...")
            domain_data[key]['formulations']['combined'].to_netcdf(out_nc)
        else:
            logger.info(f"Skipping ensemble save; using pre-computed file: {value['formulations']['ensemble_file']}")
        
        # Calculate Metrics
        domain_data[key]['metrics'] = workflow.calculate_metrics(domain_data[key], config.metrics)
        metrics_list.append(pd.DataFrame(domain_data[key]['metrics']))
        
        # Create visualizations
        # TODO: Add domain specific visualizations to reading files loop
        workflow.produce_domain_specific_visualizations(domain_data[key], config.viz, config.io, config.stats)
        
        
    metrics_df = pd.concat(metrics_list)
    if not metrics_df.empty and config.metrics.metrics_output_file:
        out_csv = config.io.output_dir / config.metrics.metrics_output_file
        metrics_df.to_csv(out_csv, index=False)
        logger.info(f"Metrics saved to {out_csv}")
    
    # Static Metric Maps (For each formulation/source)
    if config.viz.metrics_maps.enabled:
        sources = metrics_df['source'].unique()
        
        # Loop through each metric defined in the config (e.g. nse, kge, pbias)
        for metric in config.viz.metrics_maps.variables:
            
            # Loop through each formulation (e.g. 'ensemble_mean', 'noahowp_topmodel')
            for src in sources:
                src_df = metrics_df[metrics_df['source'] == src]
                
                # Check if this specific metric was actually calculated
                if metric in src_df.columns:
                    workflow.plot_metrics_on_map(
                        metrics_df=src_df,
                        variable=metric, 
                        output_path=config.io.output_dir / f"map_{metric}_{src}.png",
                        title=f"{metric.upper()} ({src})"
                    )
                else:
                    logger.warning(f"Metric '{metric}' not found in dataframe for source '{src}'. Skipping map.")

    # Interactive Map (Groups all sources together)
    if config.viz.interactive_map.enabled:
        from teval.viz.interactive import plot_interactive_metrics_map
        map_out = config.io.output_dir / "interactive_metrics_map.html"
        plot_interactive_metrics_map(metrics_df, output_path=map_out)

if __name__ == "__main__":
    main()