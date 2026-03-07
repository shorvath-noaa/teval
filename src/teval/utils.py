import geopandas as gpd
from pathlib import Path
from typing import Tuple, Optional
import logging
import time

logger = logging.getLogger(__name__)

class Timer:
    """A simple context manager for logging block execution times."""
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.start = time.time()
        print(f"\n==> STARTING: {self.name} <==")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.start
        # THE FIX: If an exception occurred, print FAILED so it's obvious!
        if exc_type is not None:
            print(f"==> FAILED: {self.name} after {elapsed:.2f} seconds ({elapsed/60:.2f} minutes) <==\n")
        else:
            print(f"==> FINISHED: {self.name} in {elapsed:.2f} seconds ({elapsed/60:.2f} minutes) <==\n")
        
def parse_run_directory(path: Path) -> Tuple[Optional[str], Optional[str]]:
    """
    Extracts (formulation_name, domain_name) from a directory name.
    
    Expected format: '{formulation}_{domain}_output'
    Example: 'sloth_noahowp_cfe_s_12009000_output'
             -> formulation: 'sloth_noahowp_cfe_s'
             -> domain: '12009000'
    """
    name = path.name
    
    # Check/Remove suffix
    if not name.endswith("_output"):
        logger.warning(f"Directory '{name}' does not end in '_output'. Skipping.")
        return None, None
    base = name[:-7] # Remove '_output'
    
    # Split by underscore
    parts = base.split('_')
    if len(parts) < 2:
        logger.warning(f"Directory '{name}' does not have enough underscore-separated parts.")
        return None, None

    # Domain is the last part, Formulation is everything else joined
    domain_name = parts[-1]
    formulation_name = "_".join(parts[:-1])
    
    return formulation_name, domain_name

def find_tailwater_feature(gdf_hydro: gpd.GeoDataFrame) -> str:
    """
    Identifies the tailwater feature in the hydrofabric as the one with no downstream connection.
    
    Args:
        gdf_hydro: The loaded hydrofabric GeoDataFrame.
        
    Returns:
        feature_id of the tailwater feature.
    """
    # Find values in toids that are not in ids
    ids = gdf_hydro['id']
    toids = gdf_hydro['toid']
    
    missing_mask = ~toids.isin(ids)
    tailwaters = gdf_hydro.loc[missing_mask].id.values
    
    # Return the list of tailwater feature_id(s)
    return tailwaters