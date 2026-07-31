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
    │     │     └── build_nexus_crosswalk()            [only when weighted]
    │     ├── weights.read_weight_file()               [only when weighted]
    │     ├── _process_formulation_files()
    │     │     ├── weights.resolve_weights()          [only when weighted]
    │     │     └── ensemble_methods.stats.build_stats()  [lazy Dask graph]
    │     └── io.observations.fetch_observations()
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
| `weights/` | Weight file reading, validation and nexus-to-feature resolution, weighting provenance |
| `metrics/` | NSE, KGE, PBIAS, significance testing |
| `viz/` | All rendering functions |
| `obs/` | USGS API observation retrieval |
| `config.py` | Pydantic models for the YAML config |
| `utils.py` | Timer, logging, timing registry |
| `experimental/` | In-development: performance-weighted mean |

## Key design decisions

**Single dask.compute() pass** — The ensemble NC write and gage-subset extraction are fused into one `dask.compute()` call. This reads each chunk of T-Route data exactly once regardless of how many downstream operations need it.

**Lazy graph construction** — `build_stats()` constructs a fully lazy Dask graph. No data is read from disk until `compute_and_write()` triggers it. Time slicing is applied to the lazy graph before compute, so only the requested period is ever read.

**Weights validated before any compute** — `teval.weights.resolve` is a pure function of plain frames, dicts and sequences: it reads no file, opens no GeoPackage and touches no Dataset. Every weight rule *and* the coverage policy are therefore decided before a Dask graph exists, so a weight file that cannot be applied fails in the first seconds of a run rather than after a long compute. Reading the (provisional) file format is `teval.weights.reader`'s separate job, so a format change leaves the rules intact.

**io/ separation** — File discovery, hydrofabric loading, and observation loading are isolated in `teval.io`. This makes each step independently testable and keeps `workflow.py` focused on compute logic.

## Data flow at CONUS scale

- ~800,000 flowpaths × 17,500 hourly timesteps × 3–4 formulations
- T-Route files are opened lazily via `xr.open_mfdataset(parallel=True)`
- Full ensemble NC is written as float32 (~several GB)
- Only the ~6,700 gage-collocated feature IDs are held in RAM for metrics/viz
