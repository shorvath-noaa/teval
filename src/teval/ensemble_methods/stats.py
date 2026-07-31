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


def _align_weights(combined_ds: xr.Dataset, weights: xr.DataArray) -> xr.DataArray:
    """
    Check a weight array against the dataset and select it onto the dataset's labels.

    xarray arithmetic joins on the *intersection* of coordinates, so a weight
    array that is missing a feature the dataset carries would silently drop
    that feature from the output rather than fail — the run would finish and
    write a product quietly short of rows.  Alignment is therefore done here,
    explicitly and up front, instead of being left to the reduction.

    The two axes are held to different standards, because they play different
    roles in the arithmetic:

    ``formulation``
        Must match the dataset exactly, as a set.  This is the axis being
        reduced, so a weight array carrying a formulation the run does not
        have would leave the selected weights summing to less than 1 and bias
        the mean low, with nothing in the result to show for it.
    every other dimension (in practice ``feature_id``)
        Must be a *superset* of the dataset's labels.  These only label rows
        of independent weight groups, so extra ones are harmless and are
        dropped; missing ones are the silent-shrink hazard above.

    Parameters
    ----------
    combined_ds:
        The lazy ensemble dataset the weights will be applied to.
    weights:
        Weight array carrying a ``formulation`` dimension, as
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
        The formulation axis is absent, unlabelled or does not match the
        dataset's, or a weight dimension omits labels the dataset carries.
    """
    if not isinstance(weights, xr.DataArray):
        raise TypeError(
            f"weights must be an xr.DataArray over (feature_id, formulation), "
            f"as resolve_weights returns; got {type(weights).__name__}."
        )
    if "formulation" not in weights.dims:
        raise ValueError(
            f"weights must carry a 'formulation' dimension to be combined over; "
            f"got dimensions {tuple(weights.dims)}."
        )
    if "formulation" not in combined_ds.coords:
        raise ValueError(
            "The ensemble dataset has no 'formulation' coordinate, so weights "
            "cannot be matched to members by name. Assign formulation names "
            "before building weighted statistics."
        )

    ds_members = pd.Index(combined_ds["formulation"].values)
    weight_members = pd.Index(weights["formulation"].values)
    unweighted = ds_members.difference(weight_members)
    unknown = weight_members.difference(ds_members)
    if len(unweighted) or len(unknown):
        raise ValueError(
            f"weights do not cover exactly the run's formulations. The run has "
            f"{list(ds_members)} and the weights carry {list(weight_members)}"
            + (f"; missing {list(unweighted)}" if len(unweighted) else "")
            + (f"; unexpected {list(unknown)}" if len(unknown) else "")
            + ". Every member must carry a weight, or the mean would be a "
            "combination of a subset that no longer sums to 1."
        )

    selection = {"formulation": ds_members.to_numpy()}
    for dim in weights.dims:
        if dim == "formulation":
            continue
        if dim not in combined_ds.coords:
            raise ValueError(
                f"weights carry a '{dim}' dimension that the ensemble dataset "
                f"does not label, so the two cannot be matched."
            )
        ds_labels = pd.Index(combined_ds[dim].values)
        absent = ds_labels.difference(pd.Index(weights[dim].values))
        if len(absent):
            shown = list(absent[:10])
            raise ValueError(
                f"weights omit {len(absent)} of the dataset's {len(ds_labels)} "
                f"'{dim}' label(s), e.g. {shown}. Weighting them would silently "
                f"drop those rows from the output instead of failing."
            )
        selection[dim] = ds_labels.to_numpy()

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
        *weights* is given but does not cover exactly the run's formulations,
        or omits a label the dataset carries on another dimension.
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