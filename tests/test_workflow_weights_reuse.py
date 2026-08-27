"""
Tests for weighting against a domain that reuses a pre-computed ensemble.

Weighting is applied in ``build_stats``, and a domain naming an
``ensemble_file`` never calls it — the cached statistics are returned as they
were written.  So a configuration that asks for weights *and* reuses an
ensemble gets neither an error nor weighted output; it silently gets whatever
the cached file already contains.  That is the one case where configured
weights go unapplied without the run failing, so it is announced with a loud
warning rather than passed over.

What is asserted here is that the warning fires exactly when it should: on a
reused ensemble with weights configured, and not on a reused ensemble without
them, not on a weighted run that actually applied its weights, and not on an
ordinary unweighted run.  ``reuses_precomputed_ensemble`` — the single
predicate all the reuse decisions ask — is tested directly alongside, including
the case of a domain map naming a file that is not on disk, which it refuses
rather than answering, so no site can act on a configuration the others would
read differently.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from teval import workflow
from teval.config import IOConfig, StatsConfig

from tests.weighting_support import (
    ALL_EQUAL_MEMBER_PARTS,
    WORKFLOW_LOGGER,
    domain_map,
    expected_mean,
    mean_of,
    weighted_config,
)


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
            domain_map(raw_files, ensemble_file=ensemble_file),
            IOConfig(),
            weighted_config(weight_file, formulation_index_map),
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
            domain_map(raw_files, ensemble_file=precomputed_ensemble_file),
            IOConfig(),
            weighted_config(weight_file, formulation_index_map),
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
            domain_map(raw_files, ensemble_file=precomputed_ensemble_file),
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
            domain_map(raw_files),
            IOConfig(),
            weighted_config(weight_file, formulation_index_map),
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
            domain_map(raw_files, ensemble_file=precomputed_ensemble_file),
            IOConfig(),
            weighted_config(weight_file, formulation_index_map),
        )

    np.testing.assert_allclose(
        results["formulations"]["combined"].streamflow_mean.compute().values, 7.0
    )
    assert len(_bypass_warnings(caplog)) == 1
    assert not [
        r for r in caplog.records if "no hydrofabric" in r.getMessage()
    ]


@pytest.mark.parametrize(
    "entry, expected",
    [
        ({"raw_files": {}, "ensemble_file": None}, False),
        ({"raw_files": {}}, False),
    ],
    ids=["no-file", "no-key"],
)
def test_the_reuse_predicate_answers_from_the_domain_map(entry, expected):
    """Naming no ensemble file is the only way not to reuse one."""
    assert workflow.reuses_precomputed_ensemble(entry) is expected


def test_the_reuse_predicate_refuses_a_named_but_absent_file():
    """
    The third answer the predicate could give is refused instead.

    Leaving it as "not a reuse" is what let ``pipeline`` and ``workflow`` drift:
    one built the statistics from raw while the other reported them
    pre-computed.  Refusing outright removes the third case, so the question is
    answerable from the domain map alone and every site gets the same answer.
    """
    with pytest.raises(FileNotFoundError, match="absent.nc"):
        workflow.reuses_precomputed_ensemble(
            {"raw_files": {}, "ensemble_file": Path("nowhere/absent.nc")}
        )


def test_the_reuse_predicate_recognises_a_file_on_disk(precomputed_ensemble_file):
    """The one case that is a reuse, so the parametrized cases mean something."""
    assert workflow.reuses_precomputed_ensemble(
        {"raw_files": {}, "ensemble_file": precomputed_ensemble_file}
    ) is True


def test_a_named_but_absent_ensemble_file_aborts_the_run(
    tmp_path, raw_files, weight_file, formulation_index_map, hydrofabric, caplog,
):
    """
    Naming an ensemble that is not there is refused, not quietly rebuilt.

    The alternative -- falling through to the raw members -- returns a mean the
    run never asked for, computed from different inputs than the configuration
    named, with nothing in the output saying so.  A path that points at nothing
    is far more likely stale than deliberate, so it stops the run and says which
    file it could not find.
    """
    with caplog.at_level(logging.DEBUG, logger=WORKFLOW_LOGGER):
        with pytest.raises(FileNotFoundError, match="never_written.nc"):
            workflow.load_domain_data(
                domain_map(raw_files, ensemble_file=tmp_path / "never_written.nc"),
                IOConfig(),
                weighted_config(weight_file, formulation_index_map),
            )

    assert _bypass_warnings(caplog) == []


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
            domain_map(raw_files), IOConfig(), StatsConfig()
        )

    ds_stats, got = mean_of(results)
    np.testing.assert_allclose(got, expected_mean(ALL_EQUAL_MEMBER_PARTS, ds_stats))
    assert not [
        r for r in caplog.records
        if r.levelno >= logging.WARNING and "weight" in r.getMessage().lower()
    ]
