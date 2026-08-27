"""
Tests for what ``load_domain_data`` does when configured weights cannot be honoured.

Three failure shapes, each with a deliberately different response:

* **Partial coverage** is normal, not an error — the weight file need not
  reach every feature — so ``on_missing="warn"`` falls back to equal weights
  and says so, while ``on_missing="error"`` aborts the domain.
* **No hydrofabric** makes the nexus-to-feature crosswalk impossible, so
  weighting cannot happen at all.  That is refused up front rather than
  quietly skipped.
* **An unreadable file or a legend naming an absent formulation** must fail
  *before* the ensemble is opened, so a misconfigured run costs a diagnostic
  rather than the whole read.

The last point is asserted by replacing ``_process_formulation_files`` with a
function that fails the test if it is called: an error raised after the
members are read is still an error, but it is a much more expensive one, and
the ordering is the thing being tested.  The per-domain summary log line is
here too, since it is the same question asked of a successful run — what did
the wiring tell the operator it did?

Expectations use the same hand-computed member parts as
``test_workflow_weights.py``.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from teval import workflow
from teval.config import IOConfig, StatsConfig

from tests.weighting_support import (
    ALL_EQUAL_MEMBER_PARTS,
    EQUAL_MEMBER_PART,
    PLAN_LOGGER,
    WEIGHTED_MEMBER_PART,
    WORKFLOW_LOGGER,
    domain_map,
    expected_mean,
    mean_of,
    weighted_config,
)


# --------------------------------------------------------------------- #
# Coverage policy through the wiring                                    #
# --------------------------------------------------------------------- #

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
            domain_map(raw_files),
            IOConfig(),
            weighted_config(partial_weight_file, formulation_index_map),
        )

    ds_stats, got = mean_of(results)
    expected = {**WEIGHTED_MEMBER_PART, 201: EQUAL_MEMBER_PART}
    np.testing.assert_allclose(got, expected_mean(expected, ds_stats))

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
            domain_map(raw_files),
            IOConfig(),
            weighted_config(
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
            domain_map(raw_files),
            IOConfig(),
            weighted_config(
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
            domain_map(raw_files),
            IOConfig(),
            weighted_config(weight_file, formulation_index_map),
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
    results = workflow.load_domain_data(domain_map(raw_files), IOConfig(), StatsConfig())

    ds_stats, got = mean_of(results)
    np.testing.assert_allclose(
        got,
        expected_mean(ALL_EQUAL_MEMBER_PARTS, ds_stats),
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
            domain_map(raw_files),
            IOConfig(),
            weighted_config(tmp_path / "absent.csv", formulation_index_map),
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
            domain_map(raw_files),
            IOConfig(),
            weighted_config(
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
    config = weighted_config(weight_file, formulation_index_map)

    with caplog.at_level(logging.DEBUG, logger=PLAN_LOGGER):
        workflow.load_domain_data(domain_map(raw_files), IOConfig(), config)

    summaries = [
        record.getMessage()
        for record in caplog.records
        if "Applying ensemble weights from" in record.getMessage()
    ]
    assert len(summaries) == 1
    assert str(weight_file) in summaries[0]
    assert "weight coverage 100.0%" in summaries[0]

    caplog.clear()
    with caplog.at_level(logging.DEBUG, logger=PLAN_LOGGER):
        workflow.load_domain_data(domain_map(raw_files), IOConfig(), config)
        workflow.load_domain_data(domain_map(raw_files), IOConfig(), config)

    repeated = [
        r for r in caplog.records
        if "Applying ensemble weights from" in r.getMessage()
    ]
    assert len(repeated) == 2


def test_no_summary_is_logged_for_an_unweighted_run(raw_files, hydrofabric, caplog):
    """Silence on the unweighted path, so the line means what it says."""
    with caplog.at_level(logging.DEBUG, logger=PLAN_LOGGER):
        workflow.load_domain_data(domain_map(raw_files), IOConfig(), StatsConfig())

    assert not [
        r for r in caplog.records if "Applying ensemble weights" in r.getMessage()
    ]
