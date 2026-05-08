"""
teval.ensemble_methods.stats

Lazy ensemble statistic computation from raw T-Route formulation files.

This module owns the xarray graph construction step only — it builds the
lazy Dask graph that describes *what* to compute but does not trigger
execution.  The actual dask.compute() call lives in pipeline.py so that the
NC write and gage extraction can be fused into a single pass.

Public API
----------
build_stats(combined_ds, raw_files, stats_config)
    Given an already-opened mfdataset and the raw file dict, return a merged
    xr.Dataset of ensemble statistics (mean, median, lower/upper spread).
"""

from __future__ import annotations

import logging
from typing import Dict

import xarray as xr

from teval.config import StatsConfig

logger = logging.getLogger(__name__)


def build_stats(
    combined_ds: xr.Dataset,
    raw_files: Dict,
    stats_config: StatsConfig,
) -> xr.Dataset:
    """
    Build a lazy xarray Dataset of ensemble statistics.

    Computes mean and median across all formulations, plus a lower/upper
    spread band.  For small ensembles (fewer members than
    ``stats_config.small_domain_threshold``) the spread is min/max; for
    larger ensembles the configured quantiles are used instead.

    All operations are lazy — no data is read from disk until the caller
    triggers ``dask.compute()``.

    Parameters
    ----------
    combined_ds:
        Lazy mfdataset with a ``formulation`` dimension produced by
        ``xr.open_mfdataset(..., concat_dim="formulation")``.
    raw_files:
        Mapping of ``{formulation_name: Path}`` used only to determine
        ensemble size (no I/O performed here).
    stats_config:
        ``StatsConfig`` instance carrying ``quantiles`` and
        ``small_domain_threshold``.

    Returns
    -------
    xr.Dataset
        Lazy dataset with variables:
        ``{var}_mean``, ``{var}_median``, ``{var}_p{lo}``, ``{var}_p{hi}``
        for every data variable in *combined_ds*.
    """
    logger.debug("Setting up lazy ensemble statistics calculations...")

    n_members = len(raw_files)
    q_lo, q_hi = stats_config.quantiles[0], stats_config.quantiles[1]

    # ------------------------------------------------------------------ #
    # Mean and median                                                      #
    # ------------------------------------------------------------------ #
    ds_mean   = combined_ds.mean(dim="formulation",   keep_attrs=True)
    ds_median = combined_ds.median(dim="formulation", keep_attrs=True)

    ds_mean   = ds_mean.rename({v: f"{v}_mean"   for v in ds_mean.data_vars})
    ds_median = ds_median.rename({v: f"{v}_median" for v in ds_median.data_vars})

    # ------------------------------------------------------------------ #
    # Spread: min/max for small ensembles, quantiles for large ones       #
    # ------------------------------------------------------------------ #
    threshold = stats_config.small_domain_threshold
    if n_members < threshold:
        logger.debug(
            f"Ensemble has {n_members} members (< threshold={threshold}); "
            "using min/max for spread band."
        )
        ds_lower = combined_ds.min(dim="formulation", keep_attrs=True)
        ds_upper = combined_ds.max(dim="formulation", keep_attrs=True)
        lo_suffix = "min"
        hi_suffix = "max"
    else:
        logger.debug(
            f"Ensemble has {n_members} members (>= threshold={threshold}); "
            f"using quantiles [{q_lo}, {q_hi}] for spread band."
        )
        ds_lower = (
            combined_ds
            .quantile(q_lo, dim="formulation", keep_attrs=True)
            .drop_vars("quantile")
        )
        ds_upper = (
            combined_ds
            .quantile(q_hi, dim="formulation", keep_attrs=True)
            .drop_vars("quantile")
        )
        lo_suffix = f"p{int(q_lo * 100):02d}"
        hi_suffix = f"p{int(q_hi * 100):02d}"

    ds_lower = ds_lower.rename({v: f"{v}_{lo_suffix}" for v in ds_lower.data_vars})
    ds_upper = ds_upper.rename({v: f"{v}_{hi_suffix}" for v in ds_upper.data_vars})

    # ------------------------------------------------------------------ #
    # Merge into single dataset                                            #
    # ------------------------------------------------------------------ #
    ds_stats = xr.merge([ds_mean, ds_median, ds_lower, ds_upper])
    ds_stats.attrs = combined_ds.attrs
    ds_stats.attrs["description"] = "Ensemble Statistics"

    return ds_stats