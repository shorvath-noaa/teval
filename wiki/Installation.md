# Installation

## Requirements

- Python >= 3.9
- A NextGen hydrofabric GeoPackage (`.gpkg`)
- T-Route NetCDF output files (one per formulation)
- USGS observations Parquet file (or enable `auto_download_usgs`)

## Steps

```bash
git clone https://github.com/shorvath-noaa/teval
cd teval
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

For development (includes pytest):
```bash
pip install -e ".[dev]"
```

## Verifying the Installation

```bash
python -m teval --help
```

## HPC Installation (Ursa)

On Ursa, load a Python module first:
```bash
module load python/3.11
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

See [Running on HPC](HPC) for Slurm job setup.
