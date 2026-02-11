import folium
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import geopandas as gpd
import xarray as xr
import pandas as pd

def map_folium(
    gdf: gpd.GeoDataFrame,
    stats_ds: xr.Dataset,  
    var_name: str = "streamflow_mean",
    time_index: int = -1,
    output_html: str = None
):
    """
    Generates an interactive Folium map.
    """
    # 1. Prepare Data
    data_slice = stats_ds.isel(time=time_index)
    raw_time = data_slice.time.values
    time_str = str(pd.to_datetime(raw_time)).replace("T", " ")
    df_data = data_slice[var_name].to_dataframe().reset_index()
    
    cols_to_keep = ['feature_id', var_name]
    df_data = df_data[[c for c in cols_to_keep if c in df_data.columns]]
    
    # Merge with Geometry
    if 'feature_id' not in gdf.columns:
        gdf = gdf.reset_index()
    gdf = gdf[['feature_id', 'geometry']]
    
    map_data = gdf.merge(df_data, on='feature_id', how='inner')
    
    if map_data.empty:
        print("Warning: Map merge resulted in empty dataset")
        return
    
    map_data['feature_id'] = map_data['feature_id'].astype(str)
    
    # Reproject if needed
    if map_data.crs != "EPSG:4326":
        map_data = map_data.to_crs("EPSG:4326")

    # 2. Color Scale
    min_val = map_data[var_name].min()
    max_val = map_data[var_name].max()
    hydro_cmap = mcolors.LinearSegmentedColormap.from_list(
        "hydro_flow", 
        ["lightskyblue", "skyblue", "deepskyblue", "dodgerblue", "blue"]
    )

    def get_color(value):
        # Normalize 0..1
        norm = (value - min_val) / (max_val - min_val) if max_val > min_val else 0
        # Get RGBA from cmap
        rgba = hydro_cmap(norm)
        # Convert to Hex for Folium
        return mcolors.to_hex(rgba)

    # 3. Initialize Map
    minx, miny, maxx, maxy = map_data.total_bounds
    center_lon = (minx + maxx) / 2
    center_lat = (miny + maxy) / 2
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
            localize=True,
            sticky=False
        )
    ).add_to(m)
    
    # Add Floating Title
    title_html = f'''
             <h3 align="center" style="font-size:16px"><b>{var_name} | {time_str}</b></h3>
             '''
    m.get_root().html.add_child(folium.Element(title_html))
    
    # Save
    if output_html:
        m.save(output_html)
        
    return m