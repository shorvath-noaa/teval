# Tests

The suite runs with `pytest` from the repository root:

```bash
pip install -e ".[dev]"
pytest
```

Shared fixtures live in `conftest.py`. They provide small synthetic stand-ins
for the inputs the ensemble machinery operates on:

| Fixture | What it provides |
| --- | --- |
| `formulation_names` | Ensemble member names, in dataset order |
| `formulation_index_map` | 1-based index to formulation name binding |
| `feature_ids` | Integer feature ids of the synthetic run |
| `combined_ds` | Lazy, dask-backed dataset over `(formulation, time, feature_id)` |
| `flowpaths_frame` | Flowpaths frame with `id` index and `toid` column, including a confluence |
| `weight_frame` | Tidy weight frame with `nexus_id`, `formulation_index`, `weight` |

Fixture values are chosen so expectations can be worked out by hand rather
than recomputed by the test.
