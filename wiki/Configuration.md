# Configuration Reference

teval is controlled entirely by a YAML configuration file. Generate a default:

```bash
python -m teval --init
```

Or view inline help for every parameter:
```bash
python -m teval --help-config
```

## Top-level sections

| Section | Description |
|---|---|
| `io` | Input/output file paths |
| `system` | CPU, Dask, stream-to-disk settings |
| `data` | Time slicing and feature filtering |
| `stats` | Ensemble statistic options |
| `metrics` | Skill metric selection |
| `viz` | All visualization flags and options |

## `io`

| Key | Type | Description |
|---|---|---|
| `troute_netcdf_dir` | path | Directory containing T-Route `*_output/` folders |
| `ensemble_netcdf_dir` | path | Pre-computed ensemble NC directory (skip recompute) |
| `hydrofabric_dir` | path | Directory containing `.gpkg` hydrofabric file(s) |
| `observations_file` | path | USGS observations Parquet or CSV file |
| `auto_download_usgs` | bool | Download observations via USGS API if file not found |
| `save_downloaded_obs` | path | Save downloaded observations to this path |
| `output_dir` | path | Root output directory |
| `per_domain_output` | bool | Create per-domain subdirectories under `output_dir` |
| `directory_naming` | `suffix`\|`parent` | T-Route directory naming convention |
| `metrics_output_file` | str | Filename for metrics CSV (default: `metrics.csv`) |

### Directory naming conventions

**`suffix`** (default) — flat layout:
```
troute_netcdf_dir/
  {formulation}_{domain}_output/
```

**`parent`** — nested layout:
```
troute_netcdf_dir/
  {domain}/
    {formulation}_output/
```

## `system`

| Key | Type | Default | Description |
|---|---|---|---|
| `cpu` | int | `-1` | Worker count. `-1` = all cores. Respects `SLURM_CPUS_PER_TASK`. |
| `stream_to_disk` | bool | `true` | Write ensemble NC during compute pass (recommended for CONUS) |
| `use_dask` | bool | `true` | Use Dask lazy evaluation |
| `logging_level` | str | `INFO` | `DEBUG`, `INFO`, `WARNING`, or `ERROR` |
| `timing` | str | `simple` | `none`, `simple`, or `verbose` |

## `stats`

| Key | Type | Default | Description |
|---|---|---|---|
| `quantiles` | list[float] | `[0.05, 0.95]` | Spread band quantiles (used when ensemble size >= `small_domain_threshold`) |
| `small_domain_threshold` | int | `10` | Below this member count, use min/max instead of quantiles |

## `metrics`

| Key | Type | Default | Description |
|---|---|---|---|
| `enabled` | bool | `true` | Compute skill metrics |
| `variables` | list | `[nse, kge, pbias]` | Metrics to compute. Options: `nse`, `kge`, `pbias`, `peak_flow_error`, `peak_timing_error` |
| `per_formulation` | bool | `true` | Compute metrics for each formulation in addition to ensemble mean |
| `bootstrap_enabled` | bool | `false` | Compute bootstrap confidence intervals (slow) |
| `bootstrap_samples` | int | `1000` | Number of bootstrap resamples |
| `confidence_level` | float | `0.95` | CI confidence level |

## `viz`

### `hydrographs`

| Key | Type | Default | Description |
|---|---|---|---|
| `enabled` | bool | `true` | Render per-gage hydrograph PNGs |
| `target_ids` | list | `[]` | Specific gage IDs to plot. Empty = all gages with observations |
| `plot_uncertainty` | bool | `false` | Shade the ensemble spread band |
| `plot_members` | bool | `true` | Plot individual formulation lines (spaghetti plot) |

### `skill_maps`

| Key | Type | Default | Description |
|---|---|---|---|
| `enabled` | bool | `true` | Render skill map figures |
| `score_maps` | bool | `true` | Spatial scatter map coloured by metric value |
| `winner_maps` | bool | `true` | Map showing which formulation wins at each gage |
| `boxplots` | bool | `true` | Score distribution boxplots per formulation |
| `vpu_breakdown` | bool | `true` | Stacked-bar win-rate by VPU (CONUS only) |
| `variables` | list | `[nse, kge, pbias]` | Metrics to map |
| `basemap` | bool | `true` | Add contextily basemap tiles |

### `interactive_map`

| Key | Type | Default | Description |
|---|---|---|---|
| `enabled` | bool | `true` | Render Folium HTML interactive map |
| `variable` | str | `streamflow_mean` | Variable to display |

### `animation`

| Key | Type | Default | Description |
|---|---|---|---|
| `enabled` | bool | `false` | Render GIF animation |
| `variable` | str | `streamflow_mean` | Variable to animate |
| `fps` | int | `8` | Frames per second |
| `log_scale` | bool | `true` | Use log colour scale |
| `cmap` | str | `hydro_flow` | Matplotlib colormap |
| `time_step` | str | `1W` | Pandas offset string for frame interval (e.g. `1D`, `1W`) |
| `min_stream_order` | int | `4` | Minimum stream order to include (reduces feature count) |
