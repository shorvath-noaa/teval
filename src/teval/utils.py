import geopandas as gpd
from shapely.geometry import Point
import teval.obs.usgs as tobs

def find_tailwater_feature(gdf_hydro: gpd.GeoDataFrame) -> str:
    """
    Identifies the tailwater feature in the hydrofabric as the one with no downstream connection.
    
    Args:
        gdf_hydro: The loaded hydrofabric GeoDataFrame.
        
    Returns:
        feature_id of the tailwater feature.
    """
    # Nexus point IDs in 'toid' that do not appear in 'id' are tailwaters.
    ids = gdf_hydro['id'].str.replace(r'\D+', '', regex=True)
    toids = gdf_hydro['toid'].str.replace(r'\D+', '', regex=True)

    # Find values in toids that are NOT in ids
    missing_mask = ~toids.isin(ids)
    tailwaters = gdf_hydro.loc[missing_mask, 'toid'].index.values
    
    # Return the list of tailwater feature_id(s)
    return tailwaters
