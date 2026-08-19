"""
Tests that configured weights reach the ensemble mean through ``load_domain_data``.

The pieces this exercises were each tested in isolation already: the reader
parses a file, the resolver enforces the rules and expands per-nexus groups
onto features, ``build_nexus_crosswalk`` derives the mapping, and
``build_stats`` applies a weight array.  What is untested until here is that
the workflow actually joins them — reads the configured file, builds the
crosswalk from *this* domain's hydrofabric, resolves against the formulation
names and feature ids the opened dataset reports, and hands the result to
``build_stats``.  A run with no weights block must come through the same code
untouched, which is asserted here too.

Expectations are hand-computed from the ``combined_ds`` fixture's closed form
(``100·member + time_step + 10·feature``) rather than from a second
implementation, so a test agrees with the code only when both are right:

* nexus 9001 carries 0.5/0.3/0.2, so its member part is
  ``0.5·100 + 0.3·200 + 0.2·300 = 170``; features 101, 102 and 103 all drain
  there (a genuine confluence, so the many-to-one expansion is exercised);
* nexus 9002 carries 0.25/0.75/0.0, so feature 201's member part is
  ``0.25·100 + 0.75·200 = 175``;
* an unweighted or equally weighted feature has member part 200.

The three are deliberately distinct, so a run that ignores weights, applies
one nexus' group to the wrong features, or transposes the formulation axis
cannot pass.

The rest of the wiring is covered alongside: what happens when the weights
cannot be applied in ``test_workflow_weights_guards.py``, the pre-computed
ensemble bypass in ``test_workflow_weights_reuse.py``, and the attributes the
run records about itself in ``test_workflow_weights_provenance.py``.
"""

from __future__ import annotations

import logging

import dask.array as da
import geopandas as gpd
import numpy as np
import pytest
import xarray as xr

from teval import workflow
from teval.config import IOConfig, StatsConfig
from teval.weights import plan as weight_plan_module

from tests.weighting_support import (
    ALL_EQUAL_MEMBER_PARTS,
    WEIGHTED_MEMBER_PART,
    domain_map,
    expected_mean,
    mean_of,
    weighted_config,
)


# --------------------------------------------------------------------- #
# The weighted path                                                     #
# --------------------------------------------------------------------- #
def test_configured_weights_reach_the_ensemble_mean(
    raw_files, weight_file, formulation_index_map, hydrofabric,
):
    """
    The end of the whole chain: a weight file on disk changes the mean.

    Nothing is stubbed between the configuration and the product, so this
    fails if any link is missing — the file unread, the crosswalk unbuilt, the
    weights unresolved, or ``build_stats`` called without them.
    """
    results = workflow.load_domain_data(
        domain_map(raw_files),
        IOConfig(),
        weighted_config(weight_file, formulation_index_map),
    )

    ds_stats, got = mean_of(results)
    np.testing.assert_allclose(got, expected_mean(WEIGHTED_MEMBER_PART, ds_stats))


def test_the_weighted_mean_is_not_the_unweighted_mean(
    raw_files, weight_file, formulation_index_map, hydrofabric,
):
    """
    Guards the test above against a coincidence.

    If the chosen weights happened to reproduce the plain mean, every
    assertion here would pass while the feature did nothing.  They do not, and
    this says so explicitly.
    """
    results = workflow.load_domain_data(
        domain_map(raw_files),
        IOConfig(),
        weighted_config(weight_file, formulation_index_map),
    )

    ds_stats, got = mean_of(results)
    unweighted = expected_mean(ALL_EQUAL_MEMBER_PARTS, ds_stats)
    assert not np.allclose(got, unweighted)


def test_every_flowpath_at_a_confluence_gets_its_nexus_weights(
    raw_files, weight_file, formulation_index_map, hydrofabric,
):
    """
    The many-to-one expansion survives the wiring.

    Features 101, 102 and 103 all drain to nexus 9001, so their means differ
    only by the fixture's per-feature offset — 10 apart — while 201, which
    drains elsewhere and carries a different group, does not follow that
    pattern.
    """
    results = workflow.load_domain_data(
        domain_map(raw_files),
        IOConfig(),
        weighted_config(weight_file, formulation_index_map),
    )

    mean = results["formulations"]["combined"].streamflow_mean
    by_feature = {
        int(f): mean.sel(feature_id=f).compute().values for f in mean.feature_id.values
    }

    np.testing.assert_allclose(by_feature[102], by_feature[101] + 10.0)
    np.testing.assert_allclose(by_feature[103], by_feature[101] + 20.0)
    # 201 sits 30 above 101 by offset alone; its different group breaks that.
    assert not np.allclose(by_feature[201], by_feature[101] + 30.0)


@pytest.fixture
def numeric_nexus_weight_file(tmp_path, weight_frame):
    """
    The same weights with the ``nexus_id`` column written unprefixed.

    ``9001`` rather than ``nex-9001``, which is a perfectly ordinary way to
    write the provisional format and which pandas reads back as a numeric
    column.
    """
    frame = weight_frame.assign(
        nexus_id=weight_frame["nexus_id"]
        .str.replace("nex-", "", regex=False)
        .astype(float)
    )
    path = tmp_path / "weights_numeric_nexus.csv"
    frame.to_csv(path, index=False)
    return path


def test_a_numeric_nexus_id_column_still_reaches_the_mean(
    raw_files, numeric_nexus_weight_file, formulation_index_map, hydrofabric, caplog,
):
    """
    A weight file whose nexus ids land as floats must weight, not fall back.

    The reader casts that column with ``astype(str)``, so the resolver sees
    ``"9001.0"``.  Reducing it by dropping non-digits gives 90010, which
    matches no nexus in the crosswalk; every feature would then take the
    equal-weight fallback and the run would finish with a coverage warning
    rather than an error, so the wrong answer would look like a wrong *file*.
    Both halves are asserted: the weights applied, and nothing warned.
    """
    with caplog.at_level(logging.WARNING):
        results = workflow.load_domain_data(
            domain_map(raw_files),
            IOConfig(),
            weighted_config(numeric_nexus_weight_file, formulation_index_map),
        )

    ds_stats, got = mean_of(results)
    np.testing.assert_allclose(got, expected_mean(WEIGHTED_MEMBER_PART, ds_stats))
    assert "uncovered" not in caplog.text


def test_build_stats_receives_the_resolved_weight_array(
    raw_files, weight_file, formulation_index_map, hydrofabric, monkeypatch,
):
    """
    What crosses the seam is a labelled array over the run's own coordinates.

    ``build_stats`` matches by label, so this pins that the workflow resolved
    against the *dataset's* formulations and feature ids rather than against
    the hydrofabric's or the raw-file dict's.
    """
    seen = {}
    real_build_stats = workflow.build_stats

    def spy(combined_ds, raw, stats_config, weights=None):
        seen["weights"] = weights
        seen["ds_features"] = [int(f) for f in combined_ds.feature_id.values]
        seen["ds_formulations"] = [str(f) for f in combined_ds.formulation.values]
        return real_build_stats(combined_ds, raw, stats_config, weights=weights)

    monkeypatch.setattr(workflow, "build_stats", spy)

    workflow.load_domain_data(
        domain_map(raw_files),
        IOConfig(),
        weighted_config(weight_file, formulation_index_map),
    )

    weights = seen["weights"]
    assert isinstance(weights, xr.DataArray)
    assert weights.dims == ("feature_id", "formulation")
    assert [int(f) for f in weights.feature_id.values] == seen["ds_features"]
    assert [str(f) for f in weights.formulation.values] == seen["ds_formulations"]
    np.testing.assert_allclose(weights.sum("formulation").values, 1.0)
    # The group for the confluence, in run order, straight off the file.
    np.testing.assert_allclose(
        weights.sel(feature_id=101).values, [0.5, 0.3, 0.2]
    )


def test_the_weighted_result_is_still_lazy(
    raw_files, weight_file, formulation_index_map, hydrofabric,
):
    """Wiring weights in must not force a compute during the load."""
    results = workflow.load_domain_data(
        domain_map(raw_files),
        IOConfig(),
        weighted_config(weight_file, formulation_index_map),
    )

    ds_stats = results["formulations"]["combined"]
    assert isinstance(ds_stats.streamflow_mean.data, da.Array)


def test_median_and_spread_are_untouched_by_wiring_weights_in(
    raw_files, weight_file, formulation_index_map, hydrofabric,
):
    """
    Only the mean moves.

    The same load run with and without the weights block must agree on
    ``_median``, ``_min`` and ``_max`` exactly, and disagree on ``_mean``.
    """
    weighted = workflow.load_domain_data(
        domain_map(raw_files),
        IOConfig(),
        weighted_config(weight_file, formulation_index_map),
    )["formulations"]["combined"]
    plain = workflow.load_domain_data(
        domain_map(raw_files), IOConfig(), StatsConfig()
    )["formulations"]["combined"]

    assert set(weighted.data_vars) == set(plain.data_vars)
    for var in ("streamflow_median", "streamflow_min", "streamflow_max"):
        np.testing.assert_array_equal(
            weighted[var].compute().values, plain[var].compute().values
        )
    assert not np.allclose(
        weighted.streamflow_mean.compute().values,
        plain.streamflow_mean.compute().values,
    )


# --------------------------------------------------------------------- #
# The unweighted path is untouched                                      #
# --------------------------------------------------------------------- #
def test_no_weights_block_builds_no_plan_and_reads_no_file(monkeypatch):
    """
    With no weights configured, neither the reader nor the crosswalk runs.

    Asserted by making both explode: the unweighted path must not reach them
    at all, rather than reaching them and discarding the result.
    """
    def explode(*args, **kwargs):
        raise AssertionError("the unweighted path touched the weight machinery")

    monkeypatch.setattr(weight_plan_module, "read_weight_file", explode)
    monkeypatch.setattr(weight_plan_module, "build_nexus_crosswalk", explode)

    assert weight_plan_module.prepare_weight_plan(
        StatsConfig(), gpd.GeoDataFrame()
    ) is None


def test_unweighted_run_hands_build_stats_no_weights(
    raw_files, hydrofabric, monkeypatch,
):
    """``weights`` arrives as None, so ``build_stats`` takes its original branch."""
    seen = {}
    real_build_stats = workflow.build_stats

    def spy(combined_ds, raw, stats_config, weights=None):
        seen["weights"] = weights
        return real_build_stats(combined_ds, raw, stats_config, weights=weights)

    monkeypatch.setattr(workflow, "build_stats", spy)

    results = workflow.load_domain_data(domain_map(raw_files), IOConfig(), StatsConfig())

    assert seen["weights"] is None
    ds_stats, got = mean_of(results)
    np.testing.assert_allclose(
        got,
        expected_mean(ALL_EQUAL_MEMBER_PARTS, ds_stats),
    )
