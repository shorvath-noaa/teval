"""
teval.ensemble_methods.stats

Lazy ensemble statistic computation from raw T-Route formulation files.

This module owns the xarray graph construction step only — it builds the
lazy Dask graph that describes *what* to compute but does not trigger
execution.  The actual dask.compute() call lives in pipeline.py so that the
NC write and gage extraction can be fused into a single pass.

Weighting
---------
When a weight array is supplied, **only the mean is weighted**.  The median
and the lower/upper spread band stay unweighted, and this is deliberate
rather than an omission.  A weighted mean is a well-defined linear
combination of the members; a weighted quantile is not — it requires a choice
of interpolation convention that changes the answer, and with the handful of
members an ensemble here carries, the min/max spread band reduces to picking
one member regardless of its weight.  Reporting them unweighted keeps them
honest as a description of the raw member spread.  A downweighted member
therefore still widens the band and still shifts the median, so the spread
around a weighted mean is not the spread of the weighted combination.

The consequence worth stating plainly: with weights configured, ``_mean`` and
``_median`` are no longer estimates of the same quantity, and the mean is not
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


def _require_matching_labels(
    axis: str,
    ds_labels: pd.Index,
    weight_labels: pd.Index,
) -> None:
    """
    Require a weight axis to carry exactly the dataset's labels, in any order.

    Equality on both axes, not something looser on either.  xarray arithmetic
    joins on the *intersection* of coordinates, so an array short of one label
    would silently drop that row and the run would finish, writing a product
    quietly short of rows; one carrying a label the run does not have was
    resolved against some other dataset.  Neither has a reconciliation worth
    guessing at, and both directions are reported together so an array wrong
    in both takes one run to diagnose.
    """
    missing = ds_labels.difference(weight_labels)
    unexpected = weight_labels.difference(ds_labels)
    if not len(missing) and not len(unexpected):
        return

    problems = []
    if len(missing):
        problems.append(
            f"weights omit {len(missing)} of the dataset's {len(ds_labels)}, "
            f"e.g. {list(missing[:10])}"
        )
    if len(unexpected):
        problems.append(
            f"weights carry {len(unexpected)} label(s) the dataset does not, "
            f"e.g. {list(unexpected[:10])}"
        )
    raise ValueError(
        f"weights do not carry exactly the dataset's '{axis}' labels: "
        + "; ".join(problems)
        + ". resolve_weights builds both axes from the dataset it is resolved "
        "against, so a mismatch means these weights are not this run's."
    )


def _align_weights(combined_ds: xr.Dataset, weights: xr.DataArray) -> xr.DataArray:
    """
    Check a weight array against the dataset and select it onto the dataset's labels.

    ``resolve_weights`` builds both of its coordinates from the very dataset
    the weights are then applied to, so in a real run the two agree by
    construction and the ``.sel`` below is a no-op.  What is asserted here is
    that the producer did what it claims — a labelled array over exactly this
    run's ``(feature_id, formulation)``, the only shape it produces — rather
    than that mismatched inputs can be reconciled.  The selection is kept so
    that matching stays by label and never by position: the ``formulation``
    axis follows directory scan order, and a silent transposition there would
    swap members.

    Parameters
    ----------
    combined_ds:
        The lazy ensemble dataset the weights will be applied to.
    weights:
        Weight array over ``(feature_id, formulation)``, as
        ``teval.weights.resolve.resolve_weights`` returns it.

    Returns
    -------
    xr.DataArray
        *weights* selected onto the dataset's own coordinate labels, so the
        subsequent reduction has nothing left to align.

    Raises
    ------
    TypeError
        *weights* is not an ``xr.DataArray``.
    ValueError
        *weights* omits either axis, the dataset does not label one of them,
        or an axis does not carry exactly the dataset's labels.
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
        _require_matching_labels(axis, ds_labels, pd.Index(weights[axis].values))
        selection[axis] = ds_labels.to_numpy()

    return weights.sel(selection)


def _weighted_mean(combined_ds: xr.Dataset, weights: xr.DataArray) -> xr.Dataset:
    """
    Combine members into a weighted mean over the formulation dimension.

    With groups that sum to 1 — which ``resolve_weights`` guarantees — this is
    the weighted sum ``Σ wᵢ·xᵢ`` over members.  It is expressed as xarray's
    weighted mean rather than as a literal ``(ds * w).sum()`` because of what
    the two do with a missing member: a bare product-and-sum treats a NaN as a
    zero contribution while still dividing by the full weight, biasing that
    timestep low without any sign of it in the output.  The weighted mean
    instead renormalizes over the members that are actually present, which is
    what the unweighted ``.mean()`` already does with ``skipna``.  Equal
    weights therefore reproduce the unweighted path exactly, gaps and all.

    Lazy: xarray expresses this as a dot product over the formulation axis, so
    the result stays dask-backed and nothing is read from disk here.  The
    weight array itself is expected to be in memory (as ``resolve_weights``
    returns it) — xarray screens it for missing values on construction, which
    would otherwise force a compute of the weights alone.
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
    either way, for the reasons set out in the module docstring.  When it is
    omitted the function behaves exactly as it did before weighting existed —
    the weighted branch is not reached and no alignment is attempted.

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
    # The median is taken over the raw members in both branches: only the
    # mean is weighted.  See the module docstring.
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