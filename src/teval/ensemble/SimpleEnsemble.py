import xarray as xr
from typing import List
from .AbstractEnsemble import Ensembler

class SimpleEnsembler(Ensembler):
    """
    Standard statistical ensembling: Mean, Median, and Quantiles.
    """
    def __init__(self, quantiles: List[float] = [0.05, 0.95]):
        self.quantiles = quantiles

    def process(self, ds_ensemble: xr.Dataset) -> xr.Dataset:
        """
        Calculates basic statistics across the ensemble member dimension.
        """
        # 1. Identify the member dimension
        dims = list(ds_ensemble.dims)
        member_dim = None
        for d in ['formulation_id', 'member', 'ensemble', 'run']:
            if d in dims:
                member_dim = d
                break
            # Case-insensitive check
            for actual_dim in dims:
                if actual_dim.lower() == d:
                    member_dim = actual_dim
                    break
        
        if member_dim is None:
            raise ValueError(f"Could not identify ensemble member dimension in {dims}")

        # 2. Calculate Stats
        print(f"   Calculating Mean/Median across '{member_dim}'...")
        ds_mean = ds_ensemble.mean(dim=member_dim, keep_attrs=True)
        ds_median = ds_ensemble.median(dim=member_dim, keep_attrs=True)
        
        # Rename variables for clarity
        ds_mean = ds_mean.rename({v: f"{v}_mean" for v in ds_mean.data_vars})
        ds_median = ds_median.rename({v: f"{v}_median" for v in ds_median.data_vars})
        
        # Merge
        ds_stats = xr.merge([ds_mean, ds_median])
        
        # 3. Calculate Quantiles
        if self.quantiles:
            print(f"   Calculating Quantiles: {self.quantiles}...")
            # xarray quantile returns a new dimension 'quantile', we want to flatten it
            ds_q = ds_ensemble.quantile(self.quantiles, dim=member_dim, keep_attrs=True)
            
            # Split out each quantile into its own variable (e.g., streamflow_p05)
            for q in self.quantiles:
                q_label = f"p{int(q*100):02d}"
                # Select the specific quantile slice
                ds_slice = ds_q.sel(quantile=q, drop=True)
                ds_slice = ds_slice.rename({v: f"{v}_{q_label}" for v in ds_slice.data_vars})
                ds_stats = xr.merge([ds_stats, ds_slice])

        # cleanup
        ds_stats.attrs = ds_ensemble.attrs
        ds_stats.attrs['description'] = 'Ensemble Statistics (SimpleEnsembler)'
        
        return ds_stats