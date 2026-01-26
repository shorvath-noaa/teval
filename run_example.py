import matplotlib.pyplot as plt
import teval.io as tio
import teval.stats as tstats
import teval.viz.static as tviz
from pathlib import Path

def main():
    print("--- TEVAL Example Runner ---")
    
    # 1. Setup Paths
    data_dir = Path("data")
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # 2. Load Data (Adjust pattern to match your actual filenames)
    print("1. Loading Data...")
    try:
        # Matches any .nc file in data/
        ds_ensemble = tio.load_ensemble(str(data_dir / "*.nc"))
        print(f"   Loaded {len(ds_ensemble.ensemble_member)} members.")
    except Exception as e:
        print(f"   Error loading data: {e}")
        return

    # 3. Calculate Statistics
    print("2. Calculating Statistics...")
    ds_stats = tstats.calculate_basics(ds_ensemble)
    print("   Stats calculated: Mean, Median, Std, p05, p95")

    # 4. Save Results
    output_file = output_dir / "ensemble_stats.nc"
    print(f"3. Saving NetCDF to {output_file}...")
    tio.save_ensemble_stats(ds_stats, output_file)

    # 5. Generate Plot
    print("4. Generating Hydrograph...")
    # Pick the first feature_id found in the data
    target_id = ds_stats.feature_id.values[0]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    tviz.hydrograph(ds_stats, feature_id=target_id, ax=ax)
    
    plot_file = output_dir / f"hydrograph_{target_id}.png"
    plt.savefig(plot_file)
    print(f"   Plot saved to {plot_file}")
    
    print("\n✅ Example run completed successfully.")

if __name__ == "__main__":
    main()