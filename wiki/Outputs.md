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
| `streamflow_mean` | Equal-weight arithmetic mean across formulations, or a weighted sum when [`stats.weights`](https://github.com/shorvath-noaa/teval/wiki/Configuration#statsweights) is configured |
| `streamflow_median` | Median across formulations |
| `streamflow_min` | Minimum across formulations (small ensembles) |
| `streamflow_max` | Maximum across formulations (small ensembles) |
| `streamflow_p05` | 5th percentile (large ensembles) |
| `streamflow_p95` | 95th percentile (large ensembles) |

Written as float32. Dimensions: `(time, feature_id)`.

### Weighting provenance

A weighted mean and an unweighted mean are the same shape, the same dtype and
the same variable name, so nothing in the values says which one a file holds.
These global attributes make the two distinguishable from the output alone,
without the run's configuration or its log.

| Attribute | Type | Description |
|---|---|---|
| `ensemble_weighting_applied` | str | `"true"` or `"false"`. Written on both branches, so a file omitting it predates weighting rather than coming from an unweighted run. |
| `ensemble_weight_file` | str | The configured weight file, as the configuration named it. Written only when weighting was applied. |
| `ensemble_weight_coverage_fraction` | float | Fraction of the domain's features that carried supplied weights; the remainder fell back to equal weights. Written only when weighting was applied. A value below 1.0 means the mean is weighted in part of the domain and plain in the rest. |

The file path and coverage fraction are omitted rather than zero-filled on an
unweighted run: a recorded coverage of `0.0` would read as "weighting was
attempted and reached nothing", which is a different outcome. Values are
strings and floats because NetCDF has no boolean attribute type.

These attributes are recorded where the statistics are built. A run that reuses
a pre-computed ensemble from `io.ensemble_netcdf_dir` skips that step, so such a
file keeps whatever attributes it was originally written with.

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
