import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import geopandas as gpd
import pandas as pd
import xarray as xr
import numpy as np
import tempfile
import shutil
from pathlib import Path
from PIL import Image
import contextily as cx


def animate_network(gdf, stats_ds, output_path, var_name="streamflow_mean", fps=8, add_basemap=True, log_scale=True):
    print("Generating animation...")
    da = stats_ds[var_name]
    data_values = da.values 
    times = da.time.values

    # Colormap Setup
    global_min, global_max = np.nanmin(data_values), np.nanmax(data_values)
    if global_min <= 0: global_min = 0.001 
    
    if log_scale:
        norm = mcolors.LogNorm(vmin=global_min, vmax=global_max)
    else:
        norm = mcolors.Normalize(vmin=global_min, vmax=global_max)
        
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "hydro_flow", ["white", "sienna", "yellow", "limegreen", "lightskyblue", "skyblue", "deepskyblue", "dodgerblue", "blue"]
    )

    gdf_sorted = gdf.to_crs(epsg=3857)
    
    # ---------------------------------------------------------
    # DYNAMIC LINEWIDTH: Scale based on bounding box width
    # ---------------------------------------------------------
    minx, miny, maxx, maxy = gdf_sorted.total_bounds
    dx_meters = maxx - minx  # Width of domain in meters
    
    if dx_meters > 3_000_000:      # CONUS scale (>3000km)
        dynamic_lw = 0.5
    elif dx_meters > 1_000_000:    # Regional scale (>1000km)
        dynamic_lw = 1.0
    elif dx_meters > 500_000:      # Large Basin scale (>500km)
        dynamic_lw = 1.5
    else:                          # Small Basin scale (<500km)
        dynamic_lw = 2.5
        
    print(f"   Domain width: {dx_meters/1000:,.0f} km -> Setting linewidth to {dynamic_lw}")
    # ---------------------------------------------------------

    temp_dir = Path(tempfile.mkdtemp())
    frame_files = []
    total_frames = len(times)
    
    for i, t in enumerate(times):
        print(f"   Rendering frame {i+1}/{total_frames}...")
        fig, ax = plt.subplots(figsize=(15, 15))
        
        current_vals = np.maximum(data_values[i, :], global_min)
        gdf_frame = gdf_sorted.copy()
        gdf_frame['val'] = current_vals

        # Plot with dynamic linewidth
        gdf_frame.plot(column='val', ax=ax, cmap=cmap, norm=norm, linewidth=dynamic_lw, alpha=0.9)
        
        if add_basemap:
            cx.add_basemap(ax, crs=gdf_sorted.crs, source=cx.providers.USGS.USTopo)
        
        ax.set_axis_off()
        time_str = pd.to_datetime(t).strftime('%Y-%m-%d %H:%M')
        ax.set_title(f"{var_name.replace('_', ' ').title()} | {time_str}", fontsize=14)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm._A = []
        fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.04, extend='both')

        frame_path = temp_dir / f"frame_{i:04d}.png"
        plt.savefig(frame_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        frame_files.append(frame_path)

    print("   Stitching frames...")
    images = [Image.open(f) for f in frame_files]
    images[0].save(
        output_path, save_all=True, append_images=images[1:], 
        duration=int(1000 / fps), loop=0, optimize=True
    )
    shutil.rmtree(temp_dir)


# def animate_network(
#     gdf: gpd.GeoDataFrame,
#     stats_ds: xr.Dataset,
#     output_path: str,
#     var_name: str = "streamflow_mean",
#     source_name: str = "Network",
#     add_basemap: bool = True,
#     fps: int = 10,
#     log_scale: bool = True,
#     cmap_name: str = "hydro_flow"
# ):
#     """
#     Generates a GIF using Frame-by-Frame stitching.
#     """
#     print(f"Generating animation for {var_name}...")

#     gdf_plot = gdf.copy()

#     if var_name not in stats_ds:
#         raise ValueError(f"Variable {var_name} not found in dataset.")
    
#     da = stats_ds[var_name]
#     if 'feature_id' not in da.dims:
#         da = da.expand_dims('feature_id')
    
#     common_ids = np.intersect1d(gdf_plot.index.values, da.feature_id.values)
#     if len(common_ids) == 0:
#         print("Error: No common feature IDs found.")
#         return

#     gdf_sorted = gdf_plot[gdf_plot.index.isin(common_ids)].sort_index()
#     da_sorted = da.sel(feature_id=common_ids).sortby('feature_id')
#     data_values = da_sorted.values 
#     times = da_sorted.time.values

#     # Setup Limits
#     valid_data = data_values[~np.isnan(data_values)]
#     global_min = valid_data.min()
#     global_max = valid_data.max()

#     if log_scale:
#         if global_min <= 0: global_min = 0.001 
#         norm = mcolors.LogNorm(vmin=global_min, vmax=global_max)
#     else:
#         norm = mcolors.Normalize(vmin=global_min, vmax=global_max)

#     # Handle Colormap
#     if cmap_name == "hydro_flow":
#         # Custom color map
#         cmap = mcolors.LinearSegmentedColormap.from_list(
#             "hydro_flow", 
#             ["white", "sienna", "yellow", "limegreen", "lightskyblue", "skyblue", "deepskyblue", "dodgerblue", "blue"]
#         )
#     else:
#         # Standard Matplotlib color map
#         try:
#             cmap = plt.get_cmap(cmap_name)
#         except ValueError:
#             print(f"Warning: Colormap '{cmap_name}' not found. Defaulting to 'coolwarm'.")
#             cmap = plt.get_cmap("coolwarm")

#     # Generate Frames
#     if add_basemap:
#         if gdf_sorted.crs and gdf_sorted.crs.to_string() != "EPSG:3857":
#             gdf_sorted = gdf_sorted.to_crs(epsg=3857)

#     temp_dir = Path(tempfile.mkdtemp())
#     frame_files = []
    
#     try:
#         total_frames = len(times)
#         for i, t in enumerate(times):
#             if i % 10 == 0:
#                 print(f"   Rendering frame {i+1}/{total_frames}...")

#             fig, ax = plt.subplots(figsize=(10, 10))
            
#             current_vals = data_values[i, :]
#             if log_scale:
#                 current_vals = np.maximum(current_vals, global_min)

#             gdf_frame = gdf_sorted.copy()
#             gdf_frame['val'] = current_vals

#             gdf_frame.plot(
#                 column='val',
#                 ax=ax,
#                 cmap=cmap,
#                 norm=norm,
#                 linewidth=3,
#                 alpha=0.9,
#                 legend=False
#             )

#             if add_basemap:
#                 try:
#                     cx.add_basemap(ax, crs=gdf_sorted.crs, source=cx.providers.USGS.USTopo)
#                 except Exception:
#                     pass
            
#             ax.set_axis_off()
            
#             try:
#                 t_str = pd.to_datetime(t).strftime('%Y-%m-%d %H:%M')
#             except:
#                 t_str = str(t)
#             ax.set_title(f"{source_name}\n{t_str}", fontsize=14)

#             sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
#             sm._A = []
#             cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.04, extend='both')
#             cbar.set_label(f"{var_name} (cms)")

#             frame_path = temp_dir / f"frame_{i:04d}.png"
#             plt.savefig(frame_path, dpi=100, bbox_inches='tight')
#             plt.close(fig)
#             frame_files.append(frame_path)

#         print(f"   Stitching frames...")
#         images = [Image.open(f) for f in frame_files]
#         duration = int(1000 / fps)
        
#         images[0].save(
#             output_path,
#             save_all=True,
#             append_images=images[1:],
#             duration=duration,
#             loop=0,
#             optimize=True
#         )
#         print(f"   Animation saved to: {output_path}")

#     except Exception as e:
#         print(f"   Error: {e}")
#     finally:
#         shutil.rmtree(temp_dir)