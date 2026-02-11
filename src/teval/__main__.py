import argparse
import sys
import logging
from teval.config import TevalConfig, generate_default_config, generate_config_help
from teval.pipeline import run_pipeline

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

    try:
        # Load and Validate
        config = TevalConfig.from_yaml(args.config)
        
        # Run
        run_pipeline(config)
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Configuration Error:\n{e}")
        sys.exit(1)

if __name__ == "__main__":
    main()