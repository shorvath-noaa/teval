import folium
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import geopandas as gpd
import xarray as xr
import pandas as pd

def map_folium(
    stats_ds: xr.Dataset, 
    gdf: gpd.GeoDataFrame, 
    var_name: str = "streamflow_mean",
    time_index: int = -1,
    cmap_name: str = 'plasma'
):
    """
    Generates an interactive Folium map.
    """
    # 1. Prepare Data
    data_slice = stats_ds.isel(time=time_index)
    df_data = data_slice[var_name].to_dataframe().reset_index()
    
    # Merge with Geometry
    if 'feature_id' not in gdf.columns:
        gdf = gdf.reset_index()
    
    map_data = gdf.merge(df_data, on='feature_id', how='inner')
    
    # Reproject if needed
    if map_data.crs != "EPSG:4326":
        map_data = map_data.to_crs("EPSG:4326")

    # 2. Color Scale
    min_val = map_data[var_name].min()
    max_val = map_data[var_name].max()
    cmap = cm.get_cmap(cmap_name)

    def get_color(value):
        norm = (value - min_val) / (max_val - min_val) if max_val > min_val else 0
        return mcolors.to_hex(cmap(norm))

    # 3. Initialize Map
    center_lat = map_data.geometry.centroid.y.mean()
    center_lon = map_data.geometry.centroid.x.mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=10, tiles="CartoDB positron")

    # 4. Add Features
    folium.GeoJson(
        map_data,
        style_function=lambda feature: {
            'color': get_color(feature['properties'][var_name]),
            'weight': 4,
            'opacity': 0.8
        },
        tooltip=folium.GeoJsonTooltip(
            fields=['feature_id', var_name],
            aliases=['ID:', f'{var_name}:'],
            localize=True
        )
    ).add_to(m)
    
    return m