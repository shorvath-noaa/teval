import xarray as xr

def calculate_basics(
    ds: xr.Dataset, 
    dim: str = "Formulation_ID",
    lower_quantile: float = 0.05, 
    upper_quantile: float = 0.95
    ) -> xr.Dataset:
    """
    Collapses the ensemble dimension into summary statistics.
    Returns a dataset with suffix vars: _mean, _median, _std, _p5, _p95
    """
    # Define aggregations
    stats = {
        'mean': ds.mean(dim=dim),
        'median': ds.median(dim=dim),
        'std': ds.std(dim=dim),
        'p05': ds.quantile(lower_quantile, dim=dim).drop_vars("quantile", errors="ignore"), 
        'p95': ds.quantile(upper_quantile, dim=dim).drop_vars("quantile", errors="ignore")
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