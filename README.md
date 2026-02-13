# teval: T-Route Ensemble Evaluation Toolkit

**teval** is a Python library and command-line tool designed to streamline the evaluation of NextGen t-route ensemble outputs. It automates reading NetCDF ensemble files, calculating statistical summaries, and generating visualizations.

## 🌟 Key Features
* **Config-Driven Workflow:** Control everything via a simple YAML configuration file.
* **Smart Caching:** Calculates ensemble statistics once, saves them to disk, and reuses them for rapid visualization.
* **Ensemble Analysis:** Automatically computes mean, median, and user-defined uncertainty quantiles (e.g., 5th/95th).
* **Rich Visualization:**
    * **Hydrographs:** Standard uncertainty bands (p05-p95) or detailed "Spaghetti Plots" of individual members.
    * **Maps:** Static choropleth maps and interactive HTML (Folium) maps.
    * **Animations:** Generate time-series GIFs of streamflow across the network.
* **USGS Integration:** Auto-fetches and caches USGS gage observations for validation metrics.

## 📦 Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/shorvath-noaa/teval
    cd teval
    ```

2.  **Create a virtual environment (Recommended):**
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    pip install .
    ```

## 🚀 Quick Start (Command Line)

The easiest way to use `teval` is through the command line interface (CLI).

**0. (Optional) Generate Sample Data:**

If you don't have your own model output yet, you can generate realistic dummy data using the included script. You must provide a valid hydrofabric GeoPackage. First, make a `data/` directory:
```bash
mkdir data
```

Then, put your geopackage in this directory and run the script to generate sample data.
```bash
# Usage: python create_dummy_data.py <path_to_gpkg>
python create_dummy_data.py data/gage_10023000.gpkg
```

This will create 10 ensemble NetCDF files in the `data/` directory.

**1. Generate a default configuration file:**
```bash
python -m teval --init
```

This creates a teval_default_config.yaml file in your directory.

**2. Edit the configuration:**
Open `teval_default_config.yaml` and adjust paths, feature IDs, and visualization settings.

**3. Run the main workflow**
```bash
python -m teval -c teval_default_config.yaml
```

**4. Need help?**

View a description of every confiuration parameter:
```bash
python -m teval --help-config
```

## 📖 Python API Usage
You can also use `teval` as a library within your own scripts or notebooks:
``` python
from teval.config import TevalConfig
from teval.pipeline import run_pipeline

# Load config
config = TevalConfig.from_yaml("teval_config.yaml")

# Override settings programmatically
config.viz.animation.enabled = True
config.viz.hydrographs.plot_members = True

# Run
run_pipeline(config)
```

## 📂 Output Structure
The tool organizes results automatically:
```bash
output/
├── ensemble_stats.nc
├── interactive_map.html
├── hydrographs/
│   ├── hydrograph_2860507.png
│   └── hydrograph_2860516.png
├── maps/
│   └── map_streamflow_mean.png
└── animation_streamflow_mean.gif
```

## 🧪 Testing
Run the test suite to ensure everything is working correctly:
```bash
pytest
```