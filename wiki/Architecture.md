# Architecture Overview

## Pipeline

```
T-Route .nc files (one per formulation)
    │
    ▼  io.discovery.initialize_domains()
    │  Build domain map: {domain → {formulations, hydrofabric, obs}}
    │
    ▼  pipeline.run_domain()  [once per domain]
    │
    ├── workflow.load_domain_data()
    │     ├── io.hydrofabric.load_hydrofabric()
    │     ├── io.observations.fetch_observations()
    │     └── _process_formulation_files()
    │           └── ensemble_methods.stats.build_stats()  [lazy Dask graph]
    │
    ├── pipeline.compute_and_write()
    │     └── dask.compute() — single pass:
    │           ├── Write full ensemble stats → *_ensemble.nc
    │           └── Extract gage subset → RAM
    │
    ├── workflow.calculate_metrics()
    │     └── metrics.deterministic.*
    │
    └── workflow.produce_domain_specific_visualizations()
          ├── viz.static.plot_hydrographs()  [parallel, joblib]
          └── viz.animation.animate_network()
    │
    ▼  pipeline.run_skill_maps()   [ProcessPoolExecutor]
    ▼  pipeline.run_interactive_map()
```

## Module responsibilities

| Module | Responsibility |
|---|---|
| `__main__.py` | CLI, config loading, top-level loop |
| `pipeline.py` | Single-domain lifecycle, dask compute, post-processing dispatch |
| `workflow.py` | Data loading, metrics calculation, per-domain viz dispatch |
| `io/` | File discovery, hydrofabric loading, observation loading |
| `ensemble_methods/` | Lazy ensemble stat graph construction |
| `metrics/` | NSE, KGE, PBIAS, significance testing |
| `viz/` | All rendering functions |
| `obs/` | USGS API observation retrieval |
| `config.py` | Pydantic models for the YAML config |
| `utils.py` | Timer, logging, timing registry |
| `experimental/` | In-development: performance-weighted mean |

## Key design decisions

**Single dask.compute() pass** — The ensemble NC write and gage-subset extraction are fused into one `dask.compute()` call. This reads each chunk of T-Route data exactly once regardless of how many downstream operations need it.

**Lazy graph construction** — `build_stats()` constructs a fully lazy Dask graph. No data is read from disk until `compute_and_write()` triggers it. Time slicing is applied to the lazy graph before compute, so only the requested period is ever read.

**io/ separation** — File discovery, hydrofabric loading, and observation loading are isolated in `teval.io`. This makes each step independently testable and keeps `workflow.py` focused on compute logic.

## Data flow at CONUS scale

- ~800,000 flowpaths × 17,500 hourly timesteps × 3–4 formulations
- T-Route files are opened lazily via `xr.open_mfdataset(parallel=True)`
- Full ensemble NC is written as float32 (~several GB)
- Only the ~6,700 gage-collocated feature IDs are held in RAM for metrics/viz
