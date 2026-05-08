"""Interactive Folium HTML map of evaluation metrics."""

import pandas as pd
import numpy as np
import folium
from folium import MacroElement
from jinja2 import Template
import matplotlib
import matplotlib.colors as mcolors
from pathlib import Path


def _get_metric_color(val: float, metric: str) -> str:
    """Helper to convert a metric value into a Hex color."""
    if pd.isna(val):
        return '#808080'  # Gray for missing data

    if metric in ['nse', 'kge']:
        norm = mcolors.Normalize(vmin=0, vmax=1)
        cmap = matplotlib.colormaps['RdYlBu']
        val = max(0, min(val, 1))
        return mcolors.to_hex(cmap(norm(val)))

    elif metric == 'pbias':
        norm = mcolors.Normalize(vmin=-50, vmax=50)
        cmap = matplotlib.colormaps['RdBu']
        val = max(-50, min(val, 50))
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

    center_lat, center_lon = df_clean['lat'].mean(), df_clean['lon'].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=5, tiles="CartoDB positron")

    grouped = df_clean.groupby(['gage_id', 'feature_id', 'lat', 'lon'])

    metric_cols = ['nse', 'kge', 'pbias', 'peak_flow_error', 'peak_timing_error']
    available_metrics = [c for c in metric_cols if c in df_clean.columns]

    if not available_metrics:
        available_metrics = ['default']

    feature_groups = {}
    for i, metric in enumerate(available_metrics):
        is_visible = (i == 0)
        fg = folium.FeatureGroup(name=f"Color by {metric.upper()}", overlay=True, show=is_visible)
        m.add_child(fg)
        feature_groups[metric] = fg

    for (gage, fid, lat, lon), group in grouped:
        html = f"<div style='font-family: Arial; font-size: 12px;'>"
        html += f"<h4 style='margin-bottom: 2px;'>Gage: {gage}</h4>"
        html += f"<b>Flowpath ID:</b> {fid}<br><hr style='margin: 5px 0px;'>"

        for _, row in group.iterrows():
            source = row.get('source', 'unknown').upper()
            html += f"<b style='color: #005A9C;'>{source}</b><br>"

            metrics_text = []
            for col in available_metrics:
                if col != 'default' and pd.notna(row.get(col, np.nan)):
                    metrics_text.append(f"{col.upper()}: {row[col]:.2f}")

            if metrics_text:
                html += " | ".join(metrics_text) + "<br>"

            if 'sig_class' in row and pd.notna(row['sig_class']):
                html += f"<i>Significance: {row['sig_class'].title()}</i><br>"

            html += "<br>"
        html += "</div>"

        mean_row = group[group['source'].str.lower() == 'ensemble_mean']
        color_row = mean_row.iloc[0] if not mean_row.empty else group.iloc[0]

        for metric in available_metrics:
            if metric == 'default':
                color = '#3186cc'
            else:
                color = _get_metric_color(color_row.get(metric, np.nan), metric)

            folium.CircleMarker(
                location=[lat, lon],
                radius=6,
                color='#333333',
                weight=1.5,
                fill=True,
                fill_color=color,
                fill_opacity=0.9,
                tooltip=folium.Tooltip(html, max_width=350)
            ).add_to(feature_groups[metric])

    folium.LayerControl(collapsed=False).add_to(m)

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
                <span>&lt;= 0 (Bad)</span>
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