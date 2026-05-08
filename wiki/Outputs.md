# Output Reference

## Directory structure

### Single domain (per_domain_output: false)
```
output_dir/
├── CONUS_ensemble.nc
├── metrics.csv
├── hydrographs/
│   ├── hydrograph_01085500.png
│   └── ...
├── skill_maps/
│   ├── map_kge_ensemble_mean.png
│   ├── map_winner_kge.png
│   ├── boxplot_kge.png
│   └── vpu_breakdown_kge.png
├── interactive_metrics_map.html
└── animation_streamflow_mean.gif
```

### Multi-domain (per_domain_output: true)
```
output_dir/
├── {domain_id}/
│   ├── {domain_id}_ensemble.nc
│   └── hydrographs/
├── metrics.csv          ← combined across all domains
├── skill_maps/
└── interactive_metrics_map.html
```

## Ensemble NetCDF (`*_ensemble.nc`)

The primary operational output. Contains ensemble statistics for every flowpath feature_id at every timestep.

| Variable | Description |
|---|---|
| `streamflow_mean` | Equal-weight arithmetic mean across formulations |
| `streamflow_median` | Median across formulations |
| `streamflow_min` | Minimum across formulations (small ensembles) |
| `streamflow_max` | Maximum across formulations (small ensembles) |
| `streamflow_p05` | 5th percentile (large ensembles) |
| `streamflow_p95` | 95th percentile (large ensembles) |

Written as float32. Dimensions: `(time, feature_id)`.

## Metrics CSV (`metrics.csv`)

One row per (gage, source) pair.

| Column | Description |
|---|---|
| `feature_id` | NHD+ feature ID of the gage flowpath |
| `gage_id` | Zero-padded 8-digit USGS gage ID |
| `lat`, `lon` | Gage location (WGS84) |
| `source` | `ensemble_mean` or formulation name |
| `nse` | Nash-Sutcliffe Efficiency |
| `kge` | Kling-Gupta Efficiency |
| `pbias` | Percent bias |

## Hydrographs

One PNG per gage with valid observations. Shows observed streamflow (black) and simulated ensemble mean (blue) with optional uncertainty band and individual formulation lines.

## Skill Maps

| File | Description |
|---|---|
| `map_{metric}_{source}.png` | Spatial scatter map, points coloured by metric value |
| `map_winner_{metric}.png` | Map showing which formulation has the best score at each gage |
| `boxplot_{metric}.png` | Distribution of metric scores per formulation |
| `vpu_breakdown_{metric}.png` | Stacked-bar chart of win rates by VPU (CONUS) |
