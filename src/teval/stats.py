import xarray as xr

def calculate_basics(ds: xr.Dataset, dim: str = "ensemble_member") -> xr.Dataset:
    """
    Collapses the ensemble dimension into summary statistics.
    Returns a dataset with suffix vars: _mean, _median, _std, _p5, _p95
    """
    print("Calculating ensemble statistics...")
    
    # Define aggregations
    stats = {
        'mean': ds.mean(dim=dim),
        'median': ds.median(dim=dim),
        'std': ds.std(dim=dim),
        'p05': ds.quantile(0.05, dim=dim).drop_vars("quantile", errors="ignore"), # 5th percentile (Lower bound)
        'p95': ds.quantile(0.95, dim=dim).drop_vars("quantile", errors="ignore")  # 95th percentile (Upper bound)
    }
    
    # Merge and rename
    stat_datasets = []
    for stat_name, stat_ds in stats.items():
        # Rename variables (e.g., streamflow -> streamflow_mean)
        rename_map = {var: f"{var}_{stat_name}" for var in stat_ds.data_vars}
        stat_datasets.append(stat_ds.rename(rename_map))
        
    final_ds = xr.merge(stat_datasets)
    final_ds.attrs = ds.attrs
    final_ds.attrs['description'] = "Ensemble summary statistics (Mean, Median, Std, 5th/95th Percentiles)"
    
    return final_ds