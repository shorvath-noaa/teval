# teval: T-Route Evaluation Toolkit

**teval** is a Python library designed to streamline the evaluation of NextGen t-route ensemble outputs. It provides tools to read massive NetCDF ensemble files, calculate statistical summaries (mean, median, uncertainty), and visualize results through static hydrographs and interactive maps.

## 🌟 Features
* **Ensemble Processing:** Efficiently calculate mean, median, standard deviation, and quantiles (5th/95th) across ensemble members.
* **Visualization:** * Static hydrographs with uncertainty bands (Matplotlib).
    * Interactive maps of the river network (Folium).
* **Data IO:** Robust handling of NetCDF ensemble outputs and USGS observation fetching.

## 📦 Installation
See [INSTALL.md](INSTALL.md) for detailed instructions.

**Quick Command:**
```bash
pip install -e .
```

## 🏃 Usage
1. **Python API**
    ```python
    import teval.io as tio
    import teval.stats as tstats
    import teval.viz.static as tviz

    # 1. Load Data
    ds = tio.load_ensemble("data/ensemble_output_*.nc")

    # 2. Calculate Stats
    ds_stats = tstats.calculate_basics(ds)

    # 3. Plot Hydrograph
    tviz.hydrograph(ds_stats, feature_id=2860507)
    ```

2. **Run the Example Script**

    We proivde a standalone script to demonstrate the full workflow:
    ```bash
    python run_example.py
    ```

## 🧪 Testing
We use `pytest` for testing. To run the suite:
```bash
pytest
```