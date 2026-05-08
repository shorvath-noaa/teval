# teval

**teval** is a Python post-processing and evaluation toolkit for [NextGen National Water Model](https://github.com/NOAA-OWP/ngen) ensemble streamflow outputs routed through [T-Route](https://github.com/NOAA-OWP/t-route). It combines multi-formulation T-Route outputs into an ensemble NetCDF product, computes skill metrics against USGS observations, and produces a suite of visualizations.

## Overview

teval sits at the end of a NextGen modeling workflow:

```
T-Route NetCDF outputs (one per formulation)
    │
    ▼  teval
    ├── ensemble_stats.nc     ← primary operational output
    ├── metrics.csv           ← KGE, NSE, PBIAS per gage per formulation
    ├── hydrographs/          ← observed vs simulated per gage
    ├── skill_maps/           ← spatial score, winner, and boxplot maps
    ├── interactive_map.html  ← Folium metric map
    └── animation.gif         ← time-lapse streamflow propagation
```

## Installation

```bash
git clone https://github.com/shorvath-noaa/teval
cd teval
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Quick Start

**1. Generate a default config file:**
```bash
python -m teval --init
```

**2. Edit `teval_default_config.yaml`** — set your input paths, output directory, and enable the outputs you want.

**3. Run:**
```bash
python -m teval -c teval_config.yaml
```

**4. View all config options:**
```bash
python -m teval --help-config
```

## Configuration

teval is entirely config-driven. A single YAML file controls all I/O paths, system options, ensemble statistics, metrics, and visualizations. See the [Configuration Wiki](https://github.com/shorvath-noaa/teval/wiki/Configuration) for a full reference.

Example configs for CONUS and multi-domain calibration runs are provided in `configs/`.

## Outputs

| Output | Description |
|---|---|
| `*_ensemble.nc` | Ensemble statistics NetCDF (mean, spread). Primary operational product. |
| `metrics.csv` | KGE, NSE, PBIAS per gage per formulation and ensemble mean. |
| `hydrographs/` | Per-gage observed vs simulated hydrograph PNGs. |
| `skill_maps/` | Score maps, winner maps, boxplots, VPU breakdown figures. |
| `interactive_metrics_map.html` | Folium interactive map of skill metrics. |
| `animation.gif` | Time-lapse GIF of streamflow propagation. |

## Two Operational Modes

**Full CONUS** — single large domain, ~800k flowpaths. Uses Dask lazy evaluation and a single-pass compute to write the ensemble NC and extract gage subsets simultaneously.

**Multi-domain calibration** — hundreds of small independent domains (one per USGS gage upstream catchment). Used for formulation calibration and ensemble method training.

## Repository Structure

```
src/teval/
├── __main__.py         CLI entry point
├── pipeline.py         Single-domain lifecycle orchestration
├── workflow.py         Data loading, metrics, per-domain visualization dispatch
├── config.py           Pydantic configuration models
├── utils.py            Timer, logging, timing registry
├── io/                 Input discovery, hydrofabric loading, observation loading
├── ensemble_methods/   Ensemble stat computation (mean, spread)
├── metrics/            NSE, KGE, PBIAS, significance testing
├── viz/                Static maps, interactive map, animation
├── obs/                USGS observation retrieval
└── experimental/       In-development features (performance-weighted mean)
```

## Documentation

Full documentation is available on the [GitHub Wiki](https://github.com/shorvath-noaa/teval/wiki):

- [Installation](https://github.com/shorvath-noaa/teval/wiki/Installation)
- [Configuration Reference](https://github.com/shorvath-noaa/teval/wiki/Configuration)
- [Architecture Overview](https://github.com/shorvath-noaa/teval/wiki/Architecture)
- [Running on HPC (Ursa)](https://github.com/shorvath-noaa/teval/wiki/HPC)
- [Output Reference](https://github.com/shorvath-noaa/teval/wiki/Outputs)

## License

CC0-1.0 — See [LICENSE](LICENSE) for details.
