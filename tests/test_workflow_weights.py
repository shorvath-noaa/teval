"""
Tests for wiring configured weights through ``load_domain_data``.

The pieces this exercises were each tested in isolation already: the reader
parses a file, the resolver enforces the rules and expands per-nexus groups
onto features, ``build_nexus_crosswalk`` derives the mapping, and
``build_stats`` applies a weight array.  What is untested until here is that
the workflow actually joins them — reads the configured file, builds the
crosswalk from *this* domain's hydrofabric, resolves against the formulation
names and feature ids the opened dataset reports, and hands the result to
``build_stats``.

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
"""

from __future__ import annotations

import logging
from pathlib import Path

import dask.array as da
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from teval import pipeline, workflow
from teval.config import IOConfig, StatsConfig, TevalConfig, WeightsConfig
from teval.weights.provenance import (
    APPLIED_ATTR,
    APPLIED_FALSE,
    APPLIED_TRUE,
    COVERAGE_ATTR,
    FILE_ATTR,
)

WORKFLOW_LOGGER = "teval.workflow"

#: The per-feature member part of the weighted mean, worked out on paper.
WEIGHTED_MEMBER_PART = {101: 170.0, 102: 170.0, 103: 170.0, 201: 175.0}
#: What a feature falling back to equal weights gets, which is the plain mean.
EQUAL_MEMBER_PART = 200.0
#: The same, for every feature -- what an unweighted run must produce.
ALL_EQUAL_MEMBER_PARTS = dict.fromkeys(WEIGHTED_MEMBER_PART, EQUAL_MEMBER_PART)


# --------------------------------------------------------------------- #
# Harness                                                               #
# --------------------------------------------------------------------- #
@pytest.fixture
def raw_files(tmp_path, combined_ds, formulation_names):
    """The synthetic ensemble written out as one NetCDF per formulation."""
    files = {}
    for name in formulation_names:
        path = tmp_path / f"{name}.nc"
        combined_ds.sel(formulation=name).drop_vars("formulation").to_netcdf(
            path, engine="h5netcdf"
        )
        files[name] = path
    return files


@pytest.fixture
def weight_file(tmp_path, weight_frame):
    """The tidy weight frame written out in the provisional csv schema."""
    path = tmp_path / "weights.csv"
    weight_frame.to_csv(path, index=False)
    return path


@pytest.fixture
def hydrofabric(monkeypatch, flowpaths_frame):
    """
    Stand in for ``load_hydrofabric`` with the synthetic flowpaths frame.

    Only the frame matters here — the crosswalk is derived from its ``toid``
    column — so the gage structures are returned empty.  Patching the loader
    rather than writing a GeoPackage keeps this about the wiring; reading a
    ``.gpkg`` is ``teval.io.hydrofabric``'s own concern.
    """
    monkeypatch.setattr(
        workflow,
        "load_hydrofabric",
        lambda gpkg_path: (flowpaths_frame, [], {}, {}),
    )
    return flowpaths_frame


@pytest.fixture
def no_hydrofabric(monkeypatch):
    """A domain with no hydrofabric at all, so no crosswalk can be built."""
    monkeypatch.setattr(
        workflow,
        "load_hydrofabric",
        lambda gpkg_path: (gpd.GeoDataFrame(), [], {}, {}),
    )


def _domain(raw_files, ensemble_file=None):
    """A domain map entry in the shape ``initialize_domains`` produces."""
    return {
        "formulations": {"raw_files": raw_files, "ensemble_file": ensemble_file},
        "hydrofabric": Path("domain.gpkg"),
        "gage_obs": {"domain_name": [], "obs_file": [None]},
    }


def _weighted_config(weight_file, formulation_index_map, **overrides):
    """A ``StatsConfig`` carrying a weights block."""
    return StatsConfig(
        weights=WeightsConfig(
            file=weight_file,
            formulation_index_map=formulation_index_map,
            **overrides,
        )
    )


def _expected_mean(member_part_by_feature, ds_stats):
    """
    Build the expected ``(time, feature_id)`` mean from the fixture's closed form.

    The whole array is reconstructed rather than a spot value, so a
    misalignment on either axis has nowhere to hide.
    """
    times = np.arange(ds_stats.sizes["time"], dtype=float)
    features = [int(f) for f in ds_stats.feature_id.values]
    feature_offset = {101: 10.0, 102: 20.0, 103: 30.0, 201: 40.0}
    return np.array(
        [
            [member_part_by_feature[f] + t + feature_offset[f] for f in features]
            for t in times
        ]
    )


def _mean_of(results):
    """The stats mean as a ``(time, feature_id)`` array."""
    ds_stats = results["formulations"]["combined"]
    return ds_stats, ds_stats.streamflow_mean.compute().transpose(
        "time", "feature_id"
    ).values


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
        _domain(raw_files),
        IOConfig(),
        _weighted_config(weight_file, formulation_index_map),
    )

    ds_stats, got = _mean_of(results)
    np.testing.assert_allclose(got, _expected_mean(WEIGHTED_MEMBER_PART, ds_stats))


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
        _domain(raw_files),
        IOConfig(),
        _weighted_config(weight_file, formulation_index_map),
    )

    ds_stats, got = _mean_of(results)
    unweighted = _expected_mean(ALL_EQUAL_MEMBER_PARTS, ds_stats)
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
        _domain(raw_files),
        IOConfig(),
        _weighted_config(weight_file, formulation_index_map),
    )

    mean = results["formulations"]["combined"].streamflow_mean
    by_feature = {
        int(f): mean.sel(feature_id=f).compute().values for f in mean.feature_id.values
    }

    np.testing.assert_allclose(by_feature[102], by_feature[101] + 10.0)
    np.testing.assert_allclose(by_feature[103], by_feature[101] + 20.0)
    # 201 sits 30 above 101 by offset alone; its different group breaks that.
    assert not np.allclose(by_feature[201], by_feature[101] + 30.0)


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
        _domain(raw_files),
        IOConfig(),
        _weighted_config(weight_file, formulation_index_map),
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
        _domain(raw_files),
        IOConfig(),
        _weighted_config(weight_file, formulation_index_map),
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
        _domain(raw_files),
        IOConfig(),
        _weighted_config(weight_file, formulation_index_map),
    )["formulations"]["combined"]
    plain = workflow.load_domain_data(
        _domain(raw_files), IOConfig(), StatsConfig()
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

    monkeypatch.setattr(workflow, "read_weight_file", explode)
    monkeypatch.setattr(workflow, "build_nexus_crosswalk", explode)

    assert workflow._prepare_weight_plan(StatsConfig(), gpd.GeoDataFrame()) is None


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

    results = workflow.load_domain_data(_domain(raw_files), IOConfig(), StatsConfig())

    assert seen["weights"] is None
    ds_stats, got = _mean_of(results)
    np.testing.assert_allclose(
        got,
        _expected_mean(ALL_EQUAL_MEMBER_PARTS, ds_stats),
    )


# --------------------------------------------------------------------- #
# Coverage policy through the wiring                                    #
# --------------------------------------------------------------------- #
@pytest.fixture
def partial_weight_file(tmp_path, weight_frame):
    """A weight file covering nexus 9001 only, leaving feature 201 uncovered."""
    path = tmp_path / "partial_weights.csv"
    weight_frame[weight_frame["nexus_id"] == "nex-9001"].to_csv(path, index=False)
    return path


def test_uncovered_features_fall_back_to_equal_weights(
    raw_files, partial_weight_file, formulation_index_map, hydrofabric, caplog,
):
    """
    Under the default 'warn', a feature whose nexus is absent gets the plain mean.

    The covered features keep the file's group, so this also pins that the
    fallback is per feature rather than an all-or-nothing retreat.
    """
    with caplog.at_level(logging.WARNING):
        results = workflow.load_domain_data(
            _domain(raw_files),
            IOConfig(),
            _weighted_config(partial_weight_file, formulation_index_map),
        )

    ds_stats, got = _mean_of(results)
    expected = {**WEIGHTED_MEMBER_PART, 201: EQUAL_MEMBER_PART}
    np.testing.assert_allclose(got, _expected_mean(expected, ds_stats))

    assert any(
        "uncovered" in record.getMessage()
        for record in caplog.records
        if record.levelno >= logging.WARNING
    )


def test_on_missing_error_aborts_the_domain(
    raw_files, partial_weight_file, formulation_index_map, hydrofabric,
):
    """``on_missing='error'`` turns the same partial file into a failed run."""
    with pytest.raises(ValueError, match="on_missing"):
        workflow.load_domain_data(
            _domain(raw_files),
            IOConfig(),
            _weighted_config(
                partial_weight_file, formulation_index_map, on_missing="error"
            ),
        )


@pytest.mark.parametrize("on_missing", ["warn", "error"])
def test_a_domain_without_a_hydrofabric_is_refused(
    raw_files, weight_file, formulation_index_map, no_hydrofabric, on_missing,
):
    """
    No hydrofabric means no crosswalk, and that is refused rather than absorbed.

    Weights are supplied per nexus and the ensemble is indexed by feature id;
    the hydrofabric is the only thing that relates the two.  Without one there
    is no join to make, so this is a broken configuration rather than a
    coverage shortfall -- and it is not left to ``on_missing``, whose default
    'warn' would otherwise complete the run with an entirely unweighted mean
    while the user believes the file they supplied was applied.  Both policies
    are exercised because the claim is that the guard does not consult them.
    """
    with pytest.raises(ValueError, match="no hydrofabric"):
        workflow.load_domain_data(
            _domain(raw_files),
            IOConfig(),
            _weighted_config(
                weight_file, formulation_index_map, on_missing=on_missing
            ),
        )


def test_the_missing_hydrofabric_guard_fires_before_the_ensemble_is_opened(
    raw_files, weight_file, formulation_index_map, no_hydrofabric, monkeypatch,
):
    """
    The refusal costs a second, not a run's worth of file opening.

    Everything the guard needs is known once the hydrofabric step has produced
    nothing, so a configuration that cannot possibly work must not first open
    every formulation file.  Asserted by making that step explode.
    """
    def explode(*args, **kwargs):
        raise AssertionError("the formulation step ran despite no hydrofabric")

    monkeypatch.setattr(workflow, "_process_formulation_files", explode)

    with pytest.raises(ValueError, match="no hydrofabric"):
        workflow.load_domain_data(
            _domain(raw_files),
            IOConfig(),
            _weighted_config(weight_file, formulation_index_map),
        )


def test_an_unweighted_run_without_a_hydrofabric_is_untouched(
    raw_files, no_hydrofabric,
):
    """
    The guard is about weights, not about hydrofabrics.

    ``hydrofabric: None`` is a supported domain entry -- a run with metrics and
    the interactive map switched off never loads one -- so an unweighted run
    must still produce the plain mean rather than meet the new error.
    """
    results = workflow.load_domain_data(_domain(raw_files), IOConfig(), StatsConfig())

    ds_stats, got = _mean_of(results)
    np.testing.assert_allclose(
        got,
        _expected_mean(ALL_EQUAL_MEMBER_PARTS, ds_stats),
    )


# --------------------------------------------------------------------- #
# Failing early                                                         #
# --------------------------------------------------------------------- #
def test_a_missing_weight_file_fails_before_the_ensemble_is_opened(
    raw_files, formulation_index_map, hydrofabric, tmp_path, monkeypatch,
):
    """
    The file is read while the hydrofabric is still the only thing loaded.

    A run configured against a path that does not exist should cost a second,
    not the time it takes to open every formulation file first.
    """
    def explode(*args, **kwargs):
        raise AssertionError("the formulation step ran despite a bad weight file")

    monkeypatch.setattr(workflow, "_process_formulation_files", explode)

    with pytest.raises(FileNotFoundError):
        workflow.load_domain_data(
            _domain(raw_files),
            IOConfig(),
            _weighted_config(tmp_path / "absent.csv", formulation_index_map),
        )


def test_a_legend_naming_an_absent_formulation_aborts(
    raw_files, weight_file, hydrofabric,
):
    """
    The binding is checked against the run, not assumed.

    A ``formulation_index_map`` that names a formulation this run does not
    carry must abort rather than weight a subset.
    """
    with pytest.raises(ValueError, match="formulation_index_map"):
        workflow.load_domain_data(
            _domain(raw_files),
            IOConfig(),
            _weighted_config(
                weight_file, {1: "formA", 2: "formB", 3: "formZ"}
            ),
        )


# --------------------------------------------------------------------- #
# The per-domain log line                                               #
# --------------------------------------------------------------------- #
def test_the_coverage_summary_is_logged_once_per_domain(
    raw_files, weight_file, formulation_index_map, hydrofabric, caplog,
):
    """
    One summary per domain, naming the file and the coverage it achieved.

    Counted rather than merely detected: a summary emitted per formulation or
    per nexus would still "appear in the log", and that is the failure this
    step exists to prevent.
    """
    config = _weighted_config(weight_file, formulation_index_map)

    with caplog.at_level(logging.DEBUG, logger=WORKFLOW_LOGGER):
        workflow.load_domain_data(_domain(raw_files), IOConfig(), config)

    summaries = [
        record.getMessage()
        for record in caplog.records
        if "Applying ensemble weights from" in record.getMessage()
    ]
    assert len(summaries) == 1
    assert str(weight_file) in summaries[0]
    assert "weight coverage 100.0%" in summaries[0]

    caplog.clear()
    with caplog.at_level(logging.DEBUG, logger=WORKFLOW_LOGGER):
        workflow.load_domain_data(_domain(raw_files), IOConfig(), config)
        workflow.load_domain_data(_domain(raw_files), IOConfig(), config)

    repeated = [
        r for r in caplog.records
        if "Applying ensemble weights from" in r.getMessage()
    ]
    assert len(repeated) == 2


def test_no_summary_is_logged_for_an_unweighted_run(raw_files, hydrofabric, caplog):
    """Silence on the unweighted path, so the line means what it says."""
    with caplog.at_level(logging.DEBUG, logger=WORKFLOW_LOGGER):
        workflow.load_domain_data(_domain(raw_files), IOConfig(), StatsConfig())

    assert not [
        r for r in caplog.records if "Applying ensemble weights" in r.getMessage()
    ]


# --------------------------------------------------------------------- #
# A pre-computed ensemble bypasses the stats builder                    #
# --------------------------------------------------------------------- #
def test_a_precomputed_ensemble_is_returned_as_written(
    tmp_path, raw_files, weight_file, formulation_index_map, hydrofabric, caplog,
):
    """
    Weighting happens in ``build_stats``, which a pre-computed ensemble skips.

    The reused file is returned exactly as it was written, and no summary is
    logged, since no weights were applied to anything.
    """
    ensemble_file = tmp_path / "ensemble.nc"
    xr.Dataset(
        {
            "streamflow_mean": (
                ("time", "feature_id"),
                np.zeros((4, 4)) + 7.0,
            )
        },
        coords={
            "time": pd.date_range("2020-01-01", periods=4, freq="h"),
            "feature_id": [101, 102, 103, 201],
        },
    ).to_netcdf(ensemble_file, engine="h5netcdf")

    with caplog.at_level(logging.DEBUG, logger=WORKFLOW_LOGGER):
        results = workflow.load_domain_data(
            _domain(raw_files, ensemble_file=ensemble_file),
            IOConfig(),
            _weighted_config(weight_file, formulation_index_map),
        )

    ds_stats = results["formulations"]["combined"]
    np.testing.assert_allclose(ds_stats.streamflow_mean.compute().values, 7.0)
    assert not [
        r for r in caplog.records if "Applying ensemble weights" in r.getMessage()
    ]


@pytest.fixture
def precomputed_ensemble_file(tmp_path):
    """A cached ensemble NetCDF, whose statistics this run did not build."""
    path = tmp_path / "precomputed_ensemble.nc"
    xr.Dataset(
        {"streamflow_mean": (("time", "feature_id"), np.zeros((4, 4)) + 7.0)},
        coords={
            "time": pd.date_range("2020-01-01", periods=4, freq="h"),
            "feature_id": [101, 102, 103, 201],
        },
    ).to_netcdf(path, engine="h5netcdf")
    return path


def _bypass_warnings(caplog):
    """Warning records announcing that configured weights went unapplied."""
    return [
        record.getMessage()
        for record in caplog.records
        if record.levelno >= logging.WARNING
        and "ENSEMBLE WEIGHTS NOT APPLIED" in record.getMessage()
    ]


def test_reusing_a_precomputed_ensemble_with_weights_warns_loudly(
    raw_files, precomputed_ensemble_file, weight_file, formulation_index_map,
    hydrofabric, caplog,
):
    """
    The bypass is announced, once, naming both files involved.

    This configuration is legal and the run succeeds, which is exactly why it
    needs saying: the mean in the output is whatever the cached file already
    held, and nothing else in the run reports that the weight file went
    unused.  The warning names the weight file (so it is clear which
    configuration was ignored) and the ensemble file (so it is clear what to
    remove to get weighting back).
    """
    with caplog.at_level(logging.DEBUG, logger=WORKFLOW_LOGGER):
        results = workflow.load_domain_data(
            _domain(raw_files, ensemble_file=precomputed_ensemble_file),
            IOConfig(),
            _weighted_config(weight_file, formulation_index_map),
        )

    warnings = _bypass_warnings(caplog)
    assert len(warnings) == 1
    assert str(weight_file) in warnings[0]
    assert str(precomputed_ensemble_file) in warnings[0]

    # And the values really are the cached ones, unweighted by this run.
    np.testing.assert_allclose(
        results["formulations"]["combined"].streamflow_mean.compute().values, 7.0
    )


def test_no_bypass_warning_when_no_weights_are_configured(
    raw_files, precomputed_ensemble_file, hydrofabric, caplog,
):
    """
    Reusing a cached ensemble is ordinary; only doing so *with weights* is not.

    Warning on every cached run would train the reader to ignore the line.
    """
    with caplog.at_level(logging.DEBUG, logger=WORKFLOW_LOGGER):
        workflow.load_domain_data(
            _domain(raw_files, ensemble_file=precomputed_ensemble_file),
            IOConfig(),
            StatsConfig(),
        )

    assert _bypass_warnings(caplog) == []


def test_no_bypass_warning_when_the_weights_are_actually_applied(
    raw_files, weight_file, formulation_index_map, hydrofabric, caplog,
):
    """A weighted run that builds its own statistics has nothing to confess."""
    with caplog.at_level(logging.DEBUG, logger=WORKFLOW_LOGGER):
        workflow.load_domain_data(
            _domain(raw_files),
            IOConfig(),
            _weighted_config(weight_file, formulation_index_map),
        )

    assert _bypass_warnings(caplog) == []


def test_a_reused_ensemble_without_a_hydrofabric_warns_rather_than_aborts(
    raw_files, precomputed_ensemble_file, weight_file, formulation_index_map,
    no_hydrofabric, caplog,
):
    """
    The two guards do not collide, and the accurate one wins.

    A run with metrics and the interactive map switched off loads no
    hydrofabric at all, so this combination arises without anyone asking for
    it.  Refusing it would abort a run that was never going to need a
    crosswalk -- nothing here consults one -- when the true complaint is that
    the weight file went unused, which is what gets said.
    """
    with caplog.at_level(logging.DEBUG, logger=WORKFLOW_LOGGER):
        results = workflow.load_domain_data(
            _domain(raw_files, ensemble_file=precomputed_ensemble_file),
            IOConfig(),
            _weighted_config(weight_file, formulation_index_map),
        )

    np.testing.assert_allclose(
        results["formulations"]["combined"].streamflow_mean.compute().values, 7.0
    )
    assert len(_bypass_warnings(caplog)) == 1
    assert not [
        r for r in caplog.records if "no hydrofabric" in r.getMessage()
    ]


def test_neither_guard_fires_on_a_normal_unweighted_run(
    raw_files, hydrofabric, caplog,
):
    """
    A run that never mentions weights hears nothing about them.

    Both guards are keyed on the weights block, so an ordinary run must reach
    its plain mean without a warning, an error, or a line naming the feature
    at all.
    """
    with caplog.at_level(logging.DEBUG, logger=WORKFLOW_LOGGER):
        results = workflow.load_domain_data(
            _domain(raw_files), IOConfig(), StatsConfig()
        )

    ds_stats, got = _mean_of(results)
    np.testing.assert_allclose(got, _expected_mean(ALL_EQUAL_MEMBER_PARTS, ds_stats))
    assert not [
        r for r in caplog.records
        if r.levelno >= logging.WARNING and "weight" in r.getMessage().lower()
    ]


# --------------------------------------------------------------------- #
# Provenance recorded in the output                                     #
# --------------------------------------------------------------------- #
def _provenance_of(results):
    """The provenance attributes carried by the statistics dataset."""
    return results["formulations"]["combined"].attrs


def test_a_weighted_run_records_its_file_and_coverage(
    raw_files, weight_file, formulation_index_map, hydrofabric,
):
    """
    A run that applied weights says so, names the file, and reports its reach.

    The file covers both nexus, so every one of the four features is covered
    and the fraction is 1.0.
    """
    results = workflow.load_domain_data(
        _domain(raw_files),
        IOConfig(),
        _weighted_config(weight_file, formulation_index_map),
    )

    attrs = _provenance_of(results)
    assert attrs[APPLIED_ATTR] == APPLIED_TRUE
    assert attrs[FILE_ATTR] == str(weight_file)
    assert attrs[COVERAGE_ATTR] == pytest.approx(1.0)


def test_the_recorded_coverage_is_the_coverage_actually_achieved(
    raw_files, partial_weight_file, formulation_index_map, hydrofabric,
):
    """
    Partial coverage is reported as such rather than as a clean 1.0.

    Three of the four features drain to nexus 9001, which the partial file
    carries; feature 201 falls back to equal weights.  This is the case the
    attribute exists for — the file holds a mean that is weighted in part of
    the domain and plain in the rest, and nothing in the values says so.
    """
    results = workflow.load_domain_data(
        _domain(raw_files),
        IOConfig(),
        _weighted_config(partial_weight_file, formulation_index_map),
    )

    attrs = _provenance_of(results)
    assert attrs[APPLIED_ATTR] == APPLIED_TRUE
    assert attrs[COVERAGE_ATTR] == pytest.approx(0.75)


def test_an_unweighted_run_records_that_weighting_was_not_applied(
    raw_files, hydrofabric,
):
    """
    The distinguishing property, asserted on a real unweighted run.

    The flag is present and negative, and no file or coverage is claimed.
    """
    results = workflow.load_domain_data(_domain(raw_files), IOConfig(), StatsConfig())

    attrs = _provenance_of(results)
    assert attrs[APPLIED_ATTR] == APPLIED_FALSE
    assert FILE_ATTR not in attrs
    assert COVERAGE_ATTR not in attrs


def test_the_two_kinds_of_run_are_distinguishable_from_attributes_alone(
    raw_files, weight_file, formulation_index_map, hydrofabric,
):
    """
    Neither run's flag matches the other's, without looking at a single value.

    Both are loaded here rather than compared against a literal, so this fails
    if the two paths ever converge on the same claim.
    """
    weighted = workflow.load_domain_data(
        _domain(raw_files),
        IOConfig(),
        _weighted_config(weight_file, formulation_index_map),
    )
    unweighted = workflow.load_domain_data(
        _domain(raw_files), IOConfig(), StatsConfig()
    )

    assert (
        _provenance_of(weighted)[APPLIED_ATTR]
        != _provenance_of(unweighted)[APPLIED_ATTR]
    )


@pytest.mark.parametrize("weighted", [True, False], ids=["weighted", "unweighted"])
def test_the_provenance_reaches_the_written_netcdf(
    tmp_path, raw_files, weight_file, formulation_index_map, hydrofabric, weighted,
):
    """
    End of the line: the attributes are in the file on disk, not just in RAM.

    Everything between ``build_stats`` and the write is exercised for real —
    ``compute_and_write`` casts to float32 and writes through h5netcdf, either
    of which could drop the dataset's attributes or reject their types.  The
    file is reopened and the values read back.
    """
    stats_config = (
        _weighted_config(weight_file, formulation_index_map)
        if weighted
        else StatsConfig()
    )
    config = TevalConfig(
        io=IOConfig(output_dir=tmp_path / "out", per_domain_output=False),
        stats=stats_config,
    )
    domain_dict = _domain(raw_files)
    domain_data = workflow.load_domain_data(domain_dict, config.io, stats_config)

    pipeline.compute_and_write("synthetic", domain_data, domain_dict, config)

    written = domain_data["formulations"]["_full_nc_path"]
    assert written is not None and written.exists()
    with xr.open_dataset(written, engine="h5netcdf") as ds_written:
        assert ds_written.attrs[APPLIED_ATTR] == (
            APPLIED_TRUE if weighted else APPLIED_FALSE
        )
        if weighted:
            assert ds_written.attrs[FILE_ATTR] == str(weight_file)
            assert ds_written.attrs[COVERAGE_ATTR] == pytest.approx(1.0)


def test_a_precomputed_ensemble_keeps_the_provenance_it_was_written_with(
    tmp_path, raw_files, weight_file, formulation_index_map, hydrofabric,
):
    """
    Reusing a file does not let this run make claims about how it was built.

    The statistics came from somewhere else, so overwriting their provenance
    with this run's configuration would attribute a weighting that never
    touched these values — and stamping 'false' on a file that genuinely was
    weighted would be worse still.
    """
    ensemble_file = tmp_path / "ensemble.nc"
    original = {APPLIED_ATTR: APPLIED_TRUE, FILE_ATTR: "elsewhere.csv"}
    xr.Dataset(
        {"streamflow_mean": (("time", "feature_id"), np.zeros((4, 4)) + 7.0)},
        coords={
            "time": pd.date_range("2020-01-01", periods=4, freq="h"),
            "feature_id": [101, 102, 103, 201],
        },
        attrs=original,
    ).to_netcdf(ensemble_file, engine="h5netcdf")

    results = workflow.load_domain_data(
        _domain(raw_files, ensemble_file=ensemble_file),
        IOConfig(),
        _weighted_config(weight_file, formulation_index_map),
    )

    attrs = _provenance_of(results)
    assert attrs[APPLIED_ATTR] == original[APPLIED_ATTR]
    assert attrs[FILE_ATTR] == original[FILE_ATTR]
    assert COVERAGE_ATTR not in attrs
