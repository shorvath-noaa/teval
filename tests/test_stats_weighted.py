"""
Tests for the weighted branch of ``teval.ensemble_methods.stats.build_stats``.

``build_stats`` is the only place a weight array ever reaches the data, so the
properties asserted here are the ones that decide whether a weighted product
is trustworthy:

* **The mean is the combination the file asked for.**  Every expectation is a
  number worked out on paper from the synthetic dataset's closed form, not a
  second implementation of the same reduction — a reimplementation would agree
  with the code even when both are wrong.
* **Absent weights change nothing.**  The unweighted path has to stay
  bit-identical to what it produced before weighting existed, or every
  existing run silently moves.
* **Only the mean is weighted.**  Median and the spread band are asserted to
  be *identical* under weights, not merely close, because the module docstring
  makes that a documented guarantee rather than an accident.
* **Nothing is computed while the graph is built.**  Asserted by making any
  scheduler start an error, which catches a compute that happened and was
  thrown away — something checking ``.chunks`` alone would miss.

The synthetic ``combined_ds`` fixture has values
``100·member + time_step + 10·feature``, so any reduction over ``formulation``
has an obvious closed form.  With members 100/200/300 and weights
0.5/0.3/0.2 the member part of the weighted mean is
``0.5·100 + 0.3·200 + 0.2·300 = 170``; under 0.25/0.75/0.0 it is ``175``; and
under equal weights it is ``200``, the plain mean.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from dask.callbacks import Callback

from teval.config import StatsConfig
from teval.ensemble_methods.stats import build_stats
from teval.weights import resolve_weights


# --------------------------------------------------------------------- #
# Helpers and local fixtures                                            #
# --------------------------------------------------------------------- #
class NoComputeAllowed(Callback):
    """
    Context manager that turns any Dask execution into a test failure.

    Registered as a Dask callback, so it fires on the scheduler starting a
    graph regardless of which entry point triggered it.  This is deliberately
    stronger than inspecting the result's chunks afterwards: a construction
    step that computed something and discarded it would still leave a
    chunked result, and would only show up as a slow, memory-hungry run in
    production.
    """

    def _start(self, dsk):
        raise AssertionError(
            "Dask executed a graph during build_stats; construction must stay lazy."
        )


def weight_array(feature_ids, formulation_names, rows) -> xr.DataArray:
    """
    Build a dense in-memory weight array over ``(feature_id, formulation)``.

    Built here from explicit per-feature rows rather than by calling
    ``resolve_weights`` so these tests pin ``build_stats`` against the weight
    array *contract* — labelled, dense, rows summing to 1 — and do not fail
    for reasons that belong to the resolver.  One test at the end does feed
    the real resolver output in, to keep the contract honest.
    """
    return xr.DataArray(
        np.asarray([rows[fid] for fid in feature_ids], dtype=float),
        dims=("feature_id", "formulation"),
        coords={
            "feature_id": list(feature_ids),
            "formulation": list(formulation_names),
        },
    )


@pytest.fixture
def raw_files(formulation_names):
    """
    The ``{name: path}`` mapping ``build_stats`` takes.

    Only its length is read — it selects the spread-band branch — so the paths
    need not exist and no I/O happens.
    """
    return {name: Path(f"{name}.nc") for name in formulation_names}


@pytest.fixture
def stats_config():
    """Defaults: three members is under the threshold, so spread is min/max."""
    return StatsConfig()


@pytest.fixture
def weights(feature_ids, formulation_names):
    """
    Weights matching the ``weight_frame`` fixture's two nexus groups.

    Features 101, 102 and 103 share the confluence group 0.5/0.3/0.2; feature
    201 carries 0.25/0.75/0.0, whose individual zero is permitted.
    """
    return weight_array(
        feature_ids,
        formulation_names,
        {
            101: [0.5, 0.3, 0.2],
            102: [0.5, 0.3, 0.2],
            103: [0.5, 0.3, 0.2],
            201: [0.25, 0.75, 0.0],
        },
    )


@pytest.fixture
def equal_weights(feature_ids, formulation_names):
    """Uniform weights: the weighted branch's answer must be the plain mean."""
    third = 1.0 / 3.0
    return weight_array(
        feature_ids, formulation_names, {fid: [third] * 3 for fid in feature_ids}
    )


def mean_values(ds_stats) -> np.ndarray:
    """The mean variable's values as ``(time, feature_id)``, computed."""
    return ds_stats["streamflow_mean"].compute().transpose("time", "feature_id").values


# --------------------------------------------------------------------- #
# The weighted mean itself                                              #
# --------------------------------------------------------------------- #
def test_weighted_mean_matches_hand_computed_expectation(
    combined_ds, raw_files, stats_config, weights
):
    """
    Every cell equals the combination worked out on paper.

    Feature 101 (offset 10) under 0.5/0.3/0.2 is ``170 + 10 + t``; 102 and 103
    share that group at offsets 20 and 30; feature 201 (offset 40) under
    0.25/0.75/0.0 is ``175 + 40 + t``.  The whole array is checked, not one
    cell, so a per-feature or per-timestep misalignment cannot hide.
    """
    ds_stats = build_stats(combined_ds, raw_files, stats_config, weights=weights)

    times = np.arange(4)[:, None]
    member_part = np.array([170.0, 170.0, 170.0, 175.0])[None, :]
    feature_part = np.array([10.0, 20.0, 30.0, 40.0])[None, :]
    expected = member_part + feature_part + times

    np.testing.assert_allclose(mean_values(ds_stats), expected)


def test_a_zero_weight_excludes_that_member_entirely(
    combined_ds, raw_files, stats_config, feature_ids, formulation_names
):
    """
    A zero weight contributes nothing, rather than being treated as missing.

    Feature 201 already carries a zero in the shared fixture; here every
    feature puts its whole weight on one member, so the mean must reproduce
    that member exactly.  Selecting a member by weight is the strongest form
    of the alignment claim: any transposition of the formulation axis lands on
    a different member and shifts the answer by 100.  The member chosen is
    ``formC`` rather than the middle one, whose values coincide with the plain
    mean and so would pass even if the weights were ignored outright.
    """
    only_formC = weight_array(
        feature_ids, formulation_names, {fid: [0.0, 0.0, 1.0] for fid in feature_ids}
    )

    ds_stats = build_stats(combined_ds, raw_files, stats_config, weights=only_formC)

    expected = combined_ds.streamflow.sel(formulation="formC").compute()
    np.testing.assert_allclose(
        mean_values(ds_stats), expected.transpose("time", "feature_id").values
    )


def test_equal_weights_reproduce_the_unweighted_mean(
    combined_ds, raw_files, stats_config, equal_weights
):
    """Uniform weights are a plain mean, so the two branches must agree."""
    weighted = build_stats(combined_ds, raw_files, stats_config, weights=equal_weights)
    unweighted = build_stats(combined_ds, raw_files, stats_config)

    np.testing.assert_allclose(mean_values(weighted), mean_values(unweighted))


def test_equal_weights_match_the_unweighted_mean_across_a_gap(
    raw_files, stats_config, feature_ids, formulation_names
):
    """
    With a member missing, equal weights still equal the unweighted mean.

    This is the case that separates a weighted mean from a literal
    ``(ds * w).sum()``: the bare product-and-sum treats the NaN as a zero
    contribution while still dividing by the full weight, biasing the timestep
    low with nothing in the output to show for it.  Here member ``formC``
    (value 310) is absent at t=0 for feature 101, so the honest answer is the
    mean of the two members present, ``(110 + 210) / 2 = 160`` — where
    dividing by the full weight regardless would give 106.67.
    """
    times = pd.date_range("2020-01-01", periods=2, freq="h")
    values = np.array(
        [
            [[110.0, 120.0], [111.0, 121.0]],
            [[210.0, 220.0], [211.0, 221.0]],
            [[np.nan, 320.0], [311.0, 321.0]],
        ]
    )
    gapped = xr.Dataset(
        {"streamflow": (("formulation", "time", "feature_id"), values)},
        coords={
            "formulation": formulation_names,
            "time": times,
            "feature_id": [101, 102],
        },
    ).chunk({"formulation": 1})
    third = 1.0 / 3.0
    uniform = weight_array(
        [101, 102], formulation_names, {101: [third] * 3, 102: [third] * 3}
    )

    weighted = build_stats(gapped, raw_files, stats_config, weights=uniform)
    unweighted = build_stats(gapped, raw_files, stats_config)

    got = mean_values(weighted)
    np.testing.assert_allclose(got, mean_values(unweighted))
    assert got[0, 0] == pytest.approx(160.0)


def test_weights_are_matched_by_label_not_by_position(
    combined_ds, raw_files, stats_config, weights
):
    """
    A weight array with both axes permuted gives the identical answer.

    Nothing downstream guarantees the resolver hands over its coordinates in
    the dataset's order, so matching has to be by name.  Permuting both axes
    at once means a positional implementation cannot accidentally pass.
    """
    shuffled = weights.isel(
        formulation=[2, 0, 1], feature_id=[3, 1, 0, 2]
    )

    in_order = build_stats(combined_ds, raw_files, stats_config, weights=weights)
    permuted = build_stats(combined_ds, raw_files, stats_config, weights=shuffled)

    np.testing.assert_array_equal(mean_values(in_order), mean_values(permuted))


# --------------------------------------------------------------------- #
# Laziness                                                              #
# --------------------------------------------------------------------- #
def test_the_weighted_graph_is_built_without_computing(
    combined_ds, raw_files, stats_config, weights
):
    """
    No scheduler runs while the weighted graph is constructed.

    ``build_stats`` only describes the work; the single ``dask.compute()``
    that fuses the NetCDF write and the gage extraction lives in the pipeline.
    A compute here would read the whole ensemble an extra time.
    """
    with NoComputeAllowed():
        ds_stats = build_stats(combined_ds, raw_files, stats_config, weights=weights)

    assert ds_stats is not None


def test_the_weighted_result_is_still_dask_backed(
    combined_ds, raw_files, stats_config, weights
):
    """Every statistic stays deferred, the mean included."""
    ds_stats = build_stats(combined_ds, raw_files, stats_config, weights=weights)

    for var in ds_stats.data_vars:
        assert hasattr(ds_stats[var].data, "dask"), f"{var} was materialised"


# --------------------------------------------------------------------- #
# The unweighted path is untouched                                      #
# --------------------------------------------------------------------- #
def test_omitting_weights_gives_the_plain_mean(combined_ds, raw_files, stats_config):
    """The default is the reduction the module performed before weighting existed."""
    ds_stats = build_stats(combined_ds, raw_files, stats_config)

    expected = combined_ds.streamflow.mean(dim="formulation").compute()
    np.testing.assert_array_equal(
        mean_values(ds_stats), expected.transpose("time", "feature_id").values
    )


def test_weights_none_is_identical_to_not_passing_weights(
    combined_ds, raw_files, stats_config
):
    """An explicit ``None`` takes the unweighted branch, not a degenerate weighted
    one."""
    implicit = build_stats(combined_ds, raw_files, stats_config)
    explicit = build_stats(combined_ds, raw_files, stats_config, weights=None)

    assert set(implicit.data_vars) == set(explicit.data_vars)
    for var in implicit.data_vars:
        np.testing.assert_array_equal(
            implicit[var].compute().values, explicit[var].compute().values
        )


def test_the_unweighted_path_does_not_touch_the_weight_alignment(
    combined_ds, raw_files, stats_config
):
    """
    Without weights, nothing about the dataset's formulation labels is checked.

    A dataset with no ``formulation`` coordinate at all still builds, which it
    could not do if the alignment guard ran unconditionally.
    """
    unlabelled = combined_ds.drop_vars("formulation")

    ds_stats = build_stats(unlabelled, raw_files, stats_config)

    assert "streamflow_mean" in ds_stats.data_vars


# --------------------------------------------------------------------- #
# Median and the spread band stay unweighted                            #
# --------------------------------------------------------------------- #
def test_median_and_min_max_band_are_unaffected_by_weights(
    combined_ds, raw_files, stats_config, weights
):
    """
    Identical, not merely close.

    The module docstring promises that only the mean is weighted, so any
    difference at all — including one that rounding would hide — is a
    behaviour change, not a tolerance question.
    """
    weighted = build_stats(combined_ds, raw_files, stats_config, weights=weights)
    unweighted = build_stats(combined_ds, raw_files, stats_config)

    for var in ("streamflow_median", "streamflow_min", "streamflow_max"):
        np.testing.assert_array_equal(
            weighted[var].compute().values,
            unweighted[var].compute().values,
            err_msg=f"{var} moved under weights but is documented as unweighted",
        )


def test_the_quantile_band_is_unaffected_by_weights(combined_ds, raw_files, weights):
    """The same guarantee on the large-ensemble branch, which uses quantiles."""
    config = StatsConfig(small_domain_threshold=2)

    weighted = build_stats(combined_ds, raw_files, config, weights=weights)
    unweighted = build_stats(combined_ds, raw_files, config)

    assert "streamflow_p05" in weighted.data_vars
    assert "streamflow_p95" in weighted.data_vars
    for var in ("streamflow_median", "streamflow_p05", "streamflow_p95"):
        np.testing.assert_array_equal(
            weighted[var].compute().values, unweighted[var].compute().values
        )


def test_a_downweighted_member_still_sets_the_band(
    combined_ds, raw_files, stats_config, feature_ids, formulation_names
):
    """
    The documented consequence, asserted rather than left as prose.

    With almost all weight on the lowest member the mean sits near it, yet the
    band is still the raw min/max over every member — so the spread around a
    weighted mean is not the spread of the weighted combination, and the mean
    is not obliged to sit centrally within it.
    """
    lopsided = weight_array(
        feature_ids, formulation_names, {fid: [0.98, 0.01, 0.01] for fid in feature_ids}
    )

    ds_stats = build_stats(combined_ds, raw_files, stats_config, weights=lopsided)

    at_101 = ds_stats.sel(feature_id=101).isel(time=0).compute()
    assert at_101["streamflow_mean"].item() == pytest.approx(113.0)
    assert at_101["streamflow_min"].item() == pytest.approx(110.0)
    assert at_101["streamflow_max"].item() == pytest.approx(310.0)
    assert at_101["streamflow_median"].item() == pytest.approx(210.0)


def test_dataset_attributes_survive_the_weighted_branch(
    combined_ds, raw_files, stats_config, weights
):
    """Provenance written onto the source dataset is not lost to the weighting."""
    ds_stats = build_stats(combined_ds, raw_files, stats_config, weights=weights)

    assert ds_stats.attrs["description"] == "Ensemble Statistics"
    assert set(ds_stats.data_vars) == set(
        build_stats(combined_ds, raw_files, stats_config).data_vars
    )


# --------------------------------------------------------------------- #
# Guard rails on the weight array                                       #
# --------------------------------------------------------------------- #
def test_weights_that_are_not_a_data_array_raise(combined_ds, raw_files, stats_config):
    """A bare array carries no labels, so it could only be matched positionally."""
    with pytest.raises(TypeError, match="must be an xr.DataArray"):
        build_stats(
            combined_ds, raw_files, stats_config, weights=np.ones((4, 3)) / 3.0
        )


def test_weights_without_a_formulation_dimension_raise(
    combined_ds, raw_files, stats_config, feature_ids
):
    """There is no axis to combine over, so the request is meaningless."""
    per_feature = xr.DataArray(
        np.ones(len(feature_ids)),
        dims=("feature_id",),
        coords={"feature_id": list(feature_ids)},
    )

    with pytest.raises(ValueError, match="'formulation' dimension"):
        build_stats(combined_ds, raw_files, stats_config, weights=per_feature)


def test_a_dataset_without_formulation_labels_raises_under_weights(
    combined_ds, raw_files, stats_config, weights
):
    """Members can only be matched by name, so unnamed members are a hard stop."""
    unlabelled = combined_ds.drop_vars("formulation")

    with pytest.raises(ValueError, match="no 'formulation' coordinate"):
        build_stats(unlabelled, raw_files, stats_config, weights=weights)


def test_weights_missing_a_formulation_raise(
    combined_ds, raw_files, stats_config, weights
):
    """An unweighted member would be dropped from a mean that still claims to be one."""
    partial = weights.sel(formulation=["formA", "formB"])

    with pytest.raises(ValueError, match="formC"):
        build_stats(combined_ds, raw_files, stats_config, weights=partial)


def test_weights_carrying_an_unknown_formulation_raise(
    combined_ds, raw_files, stats_config, feature_ids
):
    """
    An extra member is a stale file, and silently biases the mean low.

    The weights selected for the run would sum to less than 1 with nothing in
    the output to show for it, so this fails rather than renormalising behind
    the user's back.
    """
    over_wide = weight_array(
        feature_ids,
        ["formA", "formB", "formC", "formD"],
        {fid: [0.25, 0.25, 0.25, 0.25] for fid in feature_ids},
    )

    with pytest.raises(ValueError, match="formD"):
        build_stats(combined_ds, raw_files, stats_config, weights=over_wide)


def test_weights_omitting_a_feature_raise_rather_than_shrinking_the_output(
    combined_ds, raw_files, stats_config, weights
):
    """
    The silent-shrink hazard: xarray would join on the intersection.

    Left to the reduction, a weight array short of one feature would drop that
    feature's rows from the product and the run would finish successfully with
    a file quietly missing them.
    """
    short = weights.sel(feature_id=[101, 102, 103])

    with pytest.raises(ValueError, match="omit 1 of the dataset's 4"):
        build_stats(combined_ds, raw_files, stats_config, weights=short)


def test_weights_carrying_extra_feature_ids_raise(
    combined_ds, raw_files, stats_config, feature_ids, formulation_names
):
    """
    Weights covering features the run does not have are not this run's weights.

    ``resolve_weights`` expands onto the feature ids of the dataset it is
    resolved against, so a wider array cannot have come from this run — a
    weight file describing a wider domain is narrowed there, before the array
    exists.  Silently dropping the extras here would instead let a stale array
    resolved against another domain through, weighting this run with that
    domain's numbers wherever the two happened to share an id.
    """
    wider = weight_array(
        list(feature_ids) + [999],
        formulation_names,
        {
            101: [0.5, 0.3, 0.2],
            102: [0.5, 0.3, 0.2],
            103: [0.5, 0.3, 0.2],
            201: [0.25, 0.75, 0.0],
            999: [1.0, 0.0, 0.0],
        },
    )

    with pytest.raises(ValueError, match=r"'feature_id' labels.*999"):
        build_stats(combined_ds, raw_files, stats_config, weights=wider)


def test_weights_over_the_wrong_axes_entirely_raise(
    combined_ds, raw_files, stats_config, formulation_names
):
    """
    Weights must be over ``(feature_id, formulation)``, the only shape produced.

    An array indexed by some other axis is not a near miss to be reconciled
    against ``feature_id`` — it did not come from ``resolve_weights`` at all,
    so it is rejected by shape rather than by label.
    """
    by_basin = xr.DataArray(
        np.full((2, 3), 1.0 / 3.0),
        dims=("basin", "formulation"),
        coords={"basin": ["upper", "lower"], "formulation": list(formulation_names)},
    )

    with pytest.raises(ValueError, match="must carry a 'feature_id' dimension"):
        build_stats(combined_ds, raw_files, stats_config, weights=by_basin)


def test_guard_errors_are_raised_before_any_compute(
    combined_ds, raw_files, stats_config, weights
):
    """
    A bad weight array fails while the graph is still being described.

    Catching it here means a mis-specified run stops in seconds rather than
    after reading the ensemble.
    """
    short = weights.sel(feature_id=[101, 102, 103])

    with NoComputeAllowed():
        with pytest.raises(ValueError):
            build_stats(combined_ds, raw_files, stats_config, weights=short)


# --------------------------------------------------------------------- #
# The real resolver output feeds straight in                            #
# --------------------------------------------------------------------- #
def test_resolver_output_is_accepted_unadapted(
    combined_ds,
    raw_files,
    stats_config,
    weight_frame,
    formulation_index_map,
    formulation_names,
    feature_ids,
):
    """
    What ``resolve_weights`` returns is what ``build_stats`` takes.

    The two modules are tested separately against the array contract, so this
    is the one place their agreement is pinned: the resolver's output is
    passed through with no adaptation, and the result is the same
    hand-computed mean as the equivalent array built by hand.
    """
    resolved, report = resolve_weights(
        weight_frame,
        formulation_index_map,
        formulation_names,
        {9001: [101, 102, 103], 9002: [201]},
        feature_ids,
    )
    assert report.is_complete

    ds_stats = build_stats(combined_ds, raw_files, stats_config, weights=resolved)

    times = np.arange(4)[:, None]
    expected = (
        np.array([170.0, 170.0, 170.0, 175.0])[None, :]
        + np.array([10.0, 20.0, 30.0, 40.0])[None, :]
        + times
    )
    np.testing.assert_allclose(mean_values(ds_stats), expected)
