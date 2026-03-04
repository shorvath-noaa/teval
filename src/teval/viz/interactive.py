import pandas as pd
import numpy as np
import folium
from folium import MacroElement
from jinja2 import Template
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from pathlib import Path

def _get_metric_color(val: float, metric: str) -> str:
    """Helper to convert a metric value into a Hex color."""
    if pd.isna(val):
        return '#808080'  # Gray for missing data

    if metric in ['nse', 'kge']:
        # For NSE/KGE: <=0 is Red (bad), 1 is Blue (perfect)
        norm = mcolors.Normalize(vmin=0, vmax=1)
        cmap = cm.get_cmap('RdYlBu')
        val = max(0, min(val, 1))  # Clamp values for color mapping
        return mcolors.to_hex(cmap(norm(val)))
        
    elif metric == 'pbias':
        # For PBIAS: 0 is White (perfect), Negative is Red, Positive is Blue
        norm = mcolors.Normalize(vmin=-50, vmax=50)
        cmap = cm.get_cmap('RdBu') 
        val = max(-50, min(val, 50)) # Clamp between -50% and 50%
        return mcolors.to_hex(cmap(norm(val)))
        
    return '#3186cc'  # Default blue for unknown metrics


def plot_interactive_metrics_map(metrics_df: pd.DataFrame, output_path: Path):
    """
    Generates an interactive Folium map with gage locations.
    Points are colored by metric scores, and users can toggle between metrics.
    Hovering over a gage displays a tooltip with metrics for all formulations.
    """
    if metrics_df.empty or 'lat' not in metrics_df.columns or 'lon' not in metrics_df.columns:
        print("Cannot generate interactive map: missing lat/lon data or empty dataframe.")
        return

    df_clean = metrics_df.dropna(subset=['lat', 'lon'])
    if df_clean.empty:
        return

    # Center map on the mean coordinates
    center_lat, center_lon = df_clean['lat'].mean(), df_clean['lon'].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=5, tiles="CartoDB positron")

    # Group by gage to consolidate all formulations into a single tooltip
    grouped = df_clean.groupby(['gage_id', 'feature_id', 'lat', 'lon'])

    # Determine which metrics we actually have in the dataframe to color by
    metric_cols = ['nse', 'kge', 'pbias', 'peak_flow_error', 'peak_timing_error']
    available_metrics = [c for c in metric_cols if c in df_clean.columns]
    
    if not available_metrics:
        available_metrics = ['default']

    # Create a FeatureGroup layer for each metric. 
    # overlay=True makes them sit ON TOP of the basemap (Checkboxes instead of Radio Buttons)
    feature_groups = {}
    for i, metric in enumerate(available_metrics):
        # Only show the first metric by default so the map isn't cluttered
        is_visible = (i == 0) 
        fg = folium.FeatureGroup(name=f"Color by {metric.upper()}", overlay=True, show=is_visible)
        m.add_child(fg)
        feature_groups[metric] = fg

    # Populate the map
    for (gage, fid, lat, lon), group in grouped:
        # 1. Build the HTML tooltip
        html = f"<div style='font-family: Arial; font-size: 12px;'>"
        html += f"<h4 style='margin-bottom: 2px;'>Gage: {gage}</h4>"
        html += f"<b>Flowpath ID:</b> {fid}<br><hr style='margin: 5px 0px;'>"
        
        for _, row in group.iterrows():
            source = row.get('source', 'unknown').upper()
            html += f"<b style='color: #005A9C;'>{source}</b><br>"
            
            metrics_text = []
            for col in available_metrics:
                if pd.notna(row[col]):
                    metrics_text.append(f"{col.upper()}: {row[col]:.2f}")
            
            if metrics_text:
                html += " | ".join(metrics_text) + "<br>"
            
            if 'sig_class' in row and pd.notna(row['sig_class']):
                html += f"<i>Significance: {row['sig_class'].title()}</i><br>"
                
            html += "<br>"
        html += "</div>"

        # 2. Determine which row to use for the dot's color (Prefer the ensemble_mean)
        mean_row = group[group['source'].str.lower() == 'ensemble_mean']
        color_row = mean_row.iloc[0] if not mean_row.empty else group.iloc[0]

        # 3. Add a colored marker to each FeatureGroup layer
        for metric in available_metrics:
            if metric == 'default':
                color = '#3186cc'
            else:
                color = _get_metric_color(color_row.get(metric, np.nan), metric)

            folium.CircleMarker(
                location=[lat, lon],
                radius=6,
                color='#333333',     # Dark border
                weight=1.5,
                fill=True,
                fill_color=color,    # Dynamic metric color
                fill_opacity=0.9,
                tooltip=folium.Tooltip(html, max_width=350)
            ).add_to(feature_groups[metric])

    # Add the Layer Control menu to switch between metrics
    folium.LayerControl(collapsed=False).add_to(m)

    # -------------------------------------------------------------
    # Add a custom floating HTML Legend
    # -------------------------------------------------------------
    legend_html = '''
    {% macro html(this, kwargs) %}
    <div style="position: fixed; 
                bottom: 30px; left: 30px; width: 260px; height: 230px; 
                border:2px solid grey; z-index:9999; font-size:14px;
                background-color: white; opacity: 0.95; padding: 12px;
                border-radius: 5px; box-shadow: 2px 2px 5px rgba(0,0,0,0.3);">
        <b style="font-size: 15px;">Metric Color Scales</b> (Ensemble Mean)<br>
        
        <div style="margin-top: 12px;">
            <b>NSE / KGE</b> (0 to 1)<br>
            <div style="background: linear-gradient(to right, #d73027, #fdae61, #abd9e9, #4575b4); height: 12px; width: 100%; border: 1px solid #aaa; margin-top: 2px;"></div>
            <div style="display: flex; justify-content: space-between; font-size: 12px; margin-top: 2px;">
                <span><= 0 (Bad)</span>
                <span>1 (Perfect)</span>
            </div>
        </div>
        
        <div style="margin-top: 25px;">
            <b>PBIAS</b> (-50% to +50%)<br>
            <div style="background: linear-gradient(to right, #b2182b, #f7f7f7, #2166ac); height: 12px; width: 100%; border: 1px solid #aaa; margin-top: 2px;"></div>
            <div style="display: flex; justify-content: space-between; font-size: 12px; margin-top: 2px;">
                <span>-50%</span>
                <span>0%</span>
                <span>+50%</span>
            </div>
        </div>
    </div>
    {% endmacro %}
    '''
    macro = MacroElement()
    macro._template = Template(legend_html)
    m.get_root().add_child(macro)
    
    m.save(output_path)
    print(f"Interactive map saved to {output_path}")
    

# def map_folium(
#     gdf: gpd.GeoDataFrame,
#     stats_ds: xr.Dataset,  
#     var_name: str = "streamflow_mean",
#     time_index: int = -1,
#     output_html: str = None
# ):
#     """
#     Generates an interactive Folium map.
#     """
#     # 1. Prepare Data
#     data_slice = stats_ds.isel(time=time_index)
#     raw_time = data_slice.time.values
#     time_str = str(pd.to_datetime(raw_time)).replace("T", " ")
#     df_data = data_slice[var_name].to_dataframe().reset_index()
    
#     cols_to_keep = ['feature_id', var_name]
#     df_data = df_data[[c for c in cols_to_keep if c in df_data.columns]]
    
#     # Merge with Geometry
#     if 'feature_id' not in gdf.columns:
#         gdf = gdf.reset_index()
#     gdf = gdf[['feature_id', 'geometry']]
    
#     map_data = gdf.merge(df_data, on='feature_id', how='inner')
    
#     if map_data.empty:
#         print("Warning: Map merge resulted in empty dataset")
#         return
    
#     map_data['feature_id'] = map_data['feature_id'].astype(str)
    
#     # Reproject if needed
#     if map_data.crs != "EPSG:4326":
#         map_data = map_data.to_crs("EPSG:4326")

#     # 2. Color Scale
#     min_val = map_data[var_name].min()
#     max_val = map_data[var_name].max()
#     hydro_cmap = mcolors.LinearSegmentedColormap.from_list(
#         "hydro_flow", 
#         ["lightskyblue", "skyblue", "deepskyblue", "dodgerblue", "blue"]
#     )

#     def get_color(value):
#         # Normalize 0..1
#         norm = (value - min_val) / (max_val - min_val) if max_val > min_val else 0
#         # Get RGBA from cmap
#         rgba = hydro_cmap(norm)
#         # Convert to Hex for Folium
#         return mcolors.to_hex(rgba)

#     # 3. Initialize Map
#     minx, miny, maxx, maxy = map_data.total_bounds
#     center_lon = (minx + maxx) / 2
#     center_lat = (miny + maxy) / 2
#     m = folium.Map(location=[center_lat, center_lon], zoom_start=10, tiles="CartoDB positron")
    
#     # 4. Add Features
#     folium.GeoJson(
#         map_data,
#         style_function=lambda feature: {
#             'color': get_color(feature['properties'][var_name]),
#             'weight': 4,
#             'opacity': 0.8
#         },
#         tooltip=folium.GeoJsonTooltip(
#             fields=['feature_id', var_name],
#             aliases=['ID:', f'{var_name}:'],
#             localize=True,
#             sticky=False
#         )
#     ).add_to(m)
    
#     # Add Floating Title
#     title_html = f'''
#              <h3 align="center" style="font-size:16px"><b>{var_name} | {time_str}</b></h3>
#              '''
#     m.get_root().html.add_child(folium.Element(title_html))
    
#     # Save
#     if output_html:
#         m.save(output_html)
        
#     return m

# def plot_interactive_metrics_map(metrics_df: pd.DataFrame, output_path: Path):
#     """
#     Generates an interactive Folium map with gage locations.
#     Hovering over a gage displays a tooltip with metrics for all formulations.
#     """
#     if metrics_df.empty or 'lat' not in metrics_df.columns or 'lon' not in metrics_df.columns:
#         print("Cannot generate interactive map: missing lat/lon data or empty dataframe.")
#         return

#     # Drop rows without valid coordinates
#     df_clean = metrics_df.dropna(subset=['lat', 'lon'])
#     if df_clean.empty:
#         return

#     # Center map on the mean coordinates
#     center_lat = df_clean['lat'].mean()
#     center_lon = df_clean['lon'].mean()
#     m = folium.Map(location=[center_lat, center_lon], zoom_start=5, tiles="CartoDB positron")

#     # Group by gage to consolidate all formulations into a single tooltip
#     grouped = df_clean.groupby(['gage_id', 'feature_id', 'lat', 'lon'])

#     # Standard metrics to look for
#     metric_cols = ['nse', 'kge', 'pbias', 'peak_flow_error', 'peak_timing_error']

#     for (gage, fid, lat, lon), group in grouped:
#         # Build the HTML for the tooltip
#         html = f"<div style='font-family: Arial; font-size: 12px;'>"
#         html += f"<h4 style='margin-bottom: 2px;'>Gage: {gage}</h4>"
#         html += f"<b>Flowpath ID:</b> {fid}<br><hr style='margin: 5px 0px;'>"
        
#         # Add a section for each formulation/source
#         for _, row in group.iterrows():
#             source = row.get('source', 'unknown').upper()
#             html += f"<b style='color: #005A9C;'>{source}</b><br>"
            
#             # Gather available metrics for this source
#             metrics_text = []
#             for col in metric_cols:
#                 if col in row and pd.notna(row[col]):
#                     metrics_text.append(f"{col.upper()}: {row[col]:.2f}")
            
#             if metrics_text:
#                 html += " | ".join(metrics_text) + "<br>"
                
#             # Add significance class if it exists
#             if 'sig_class' in row and pd.notna(row['sig_class']):
#                 html += f"<i>Significance: {row['sig_class'].title()}</i><br>"
                
#             html += "<br>"
            
#         html += "</div>"

#         # Add the marker to the map
#         folium.CircleMarker(
#             location=[lat, lon],
#             radius=6,
#             color='#3186cc',
#             weight=1.5,
#             fill=True,
#             fill_color='#3186cc',
#             fill_opacity=0.7,
#             tooltip=folium.Tooltip(html, max_width=350)
#         ).add_to(m)

#     # Save to file
#     m.save(output_path)
#     print(f"Interactive map saved to {output_path}")