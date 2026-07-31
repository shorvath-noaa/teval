"""
teval.ensemble_methods.stats

Lazy ensemble statistic computation from raw T-Route formulation files.

This module owns the xarray graph construction step only — it builds the
lazy Dask graph that describes *what* to compute but does not trigger
execution.  The actual dask.compute() call lives in pipeline.py so that the
NC write and gage extraction can be fused into a single pass.

Weighting
---------
When a weight array is supplied, **only the mean is weighted**; the median and
the lower/upper spread band stay unweighted, deliberately.  A weighted mean is
a well-defined linear combination of the members; a weighted quantile is not —
it needs an interpolation convention that changes the answer, and with the
handful of members an ensemble here carries, the min/max band reduces to
picking one member regardless of its weight.  Unweighted, they stay honest as
a description of the raw member spread.

The consequence worth stating plainly: with weights configured, a downweighted
member still widens the band and still shifts the median, so ``_mean`` and
``_median`` are no longer estimates of the same quantity and the mean is not
guaranteed to sit inside the spread band.

Public API
----------
build_stats(combined_ds, raw_files, stats_config, weights=None)
    Given an already-opened mfdataset and the raw file dict, return a merged
    xr.Dataset of ensemble statistics (mean, median, lower/upper spread).
    With *weights*, the mean becomes a weighted combination over the
    formulation dimension.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import pandas as pd
import xarray as xr

from teval.config import StatsConfig

logger = logging.getLogger(__name__)


def _align_weights(combined_ds: xr.Dataset, weights: xr.DataArray) -> xr.DataArray:
    """
    Require a weight array over exactly this run's labels, and select it onto them.

    Each axis must match the dataset's labels exactly, in either direction,
    because xarray arithmetic joins on the *intersection* of coordinates: an
    array short of one label would silently drop that row and the run would
    finish, writing a product quietly short of rows.  ``resolve_weights``
    builds both coordinates from the very dataset the weights are applied to,
    so in a real run they agree and the ``.sel`` is a no-op; it is kept so that
    matching stays by label and never by position, since the ``formulation``
    axis follows directory scan order and a silent transposition there would
    swap members.
    """
    if not isinstance(weights, xr.DataArray):
        raise TypeError(
            f"weights must be an xr.DataArray over (feature_id, formulation), "
            f"as resolve_weights returns; got {type(weights).__name__}."
        )

    selection = {}
    for axis in ("formulation", "feature_id"):
        if axis not in weights.dims:
            raise ValueError(
                f"weights must carry a '{axis}' dimension — resolve_weights "
                f"returns them over (feature_id, formulation); got dimensions "
                f"{tuple(weights.dims)}."
            )
        if axis not in combined_ds.coords:
            raise ValueError(
                f"The ensemble dataset has no '{axis}' coordinate, so weights "
                f"cannot be matched to it by label. Assign {axis} labels "
                f"before building weighted statistics."
            )
        ds_labels = pd.Index(combined_ds[axis].values)
        weight_labels = pd.Index(weights[axis].values)
        missing = ds_labels.difference(weight_labels)
        unexpected = weight_labels.difference(ds_labels)
        if len(missing) or len(unexpected):
            problems = []
            if len(missing):
                problems.append(
                    f"weights omit {len(missing)} of the dataset's "
                    f"{len(ds_labels)}, e.g. {list(missing[:10])}"
                )
            if len(unexpected):
                problems.append(
                    f"weights carry {len(unexpected)} label(s) the dataset "
                    f"does not, e.g. {list(unexpected[:10])}"
                )
            raise ValueError(
                f"weights do not carry exactly the dataset's '{axis}' labels: "
                + "; ".join(problems)
                + ". resolve_weights builds both axes from the dataset it is "
                "resolved against, so a mismatch means these weights are not "
                "this run's."
            )
        selection[axis] = ds_labels.to_numpy()

    return weights.sel(selection)


def _weighted_mean(combined_ds: xr.Dataset, weights: xr.DataArray) -> xr.Dataset:
    """
    Combine members into a weighted mean over the formulation dimension.

    With groups that sum to 1 — which ``resolve_weights`` guarantees — this is
    the weighted sum ``Σ wᵢ·xᵢ`` over members.  It is xarray's weighted mean
    rather than a literal ``(ds * w).sum()`` because of what the two do with a
    missing member: a bare product-and-sum treats a NaN as a zero contribution
    while still dividing by the full weight, biasing that timestep low with no
    sign of it in the output.  The weighted mean renormalizes over the members
    actually present, as the unweighted ``.mean()`` already does with
    ``skipna``, so equal weights reproduce the unweighted path exactly.

    Lazy: a dot product over the formulation axis, dask-backed, nothing read
    from disk.  The weight array itself is expected to be in memory, as
    ``resolve_weights`` returns it — xarray screens it for missing values on
    construction, which would otherwise force a compute of the weights alone.
    """
    aligned = _align_weights(combined_ds, weights)
    logger.debug(
        f"Applying weights over {aligned.sizes['formulation']} formulation(s) "
        f"to the ensemble mean; median and spread band remain unweighted."
    )
    return combined_ds.weighted(aligned).mean(dim="formulation", keep_attrs=True)


def build_stats(
    combined_ds: xr.Dataset,
    raw_files: Dict,
    stats_config: StatsConfig,
    weights: Optional[xr.DataArray] = None,
) -> xr.Dataset:
    """
    Build a lazy xarray Dataset of ensemble statistics.

    Computes mean and median across all formulations, plus a lower/upper
    spread band.  For small ensembles (fewer members than
    ``stats_config.small_domain_threshold``) the spread is min/max; for
    larger ensembles the configured quantiles are used instead.

    When *weights* is given the mean becomes a weighted combination over the
    formulation dimension; the median and the spread band are unweighted
    either way, for the reasons the module docstring sets out.  Omitted, the
    weighted branch is not reached and no alignment is attempted.

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
    weights:
        Optional weight array over ``(feature_id, formulation)`` with groups
        summing to 1, as ``teval.weights.resolve.resolve_weights`` returns it.
        Matched to the dataset by coordinate *label*, not by position, so the
        formulation axis cannot be silently transposed.  ``None`` (the
        default) means an unweighted mean.

    Returns
    -------
    xr.Dataset
        Lazy dataset with variables:
        ``{var}_mean``, ``{var}_median``, ``{var}_p{lo}``, ``{var}_p{hi}``
        for every data variable in *combined_ds*.

    Raises
    ------
    TypeError
        *weights* is given but is not an ``xr.DataArray``.
    ValueError
        *weights* is given but is not over exactly this run's ``feature_id``
        and ``formulation`` labels.
    """
    logger.debug("Setting up lazy ensemble statistics calculations...")

    n_members = len(raw_files)
    q_lo, q_hi = stats_config.quantiles[0], stats_config.quantiles[1]

    # ------------------------------------------------------------------ #
    # Mean and median                                                      #
    # ------------------------------------------------------------------ #
    if weights is None:
        ds_mean = combined_ds.mean(dim="formulation", keep_attrs=True)
    else:
        ds_mean = _weighted_mean(combined_ds, weights)

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