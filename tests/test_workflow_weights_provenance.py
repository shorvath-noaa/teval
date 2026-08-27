"""
Tests for the weighting provenance a run records about itself.

``teval.weights.provenance`` builds the attributes and is tested on its own in
``test_weights_provenance.py``; what matters here is that a real run attaches
them, that the recorded coverage is the coverage *actually achieved* rather
than the coverage requested, and that they survive the NetCDF write.

The load-bearing property is the last one in this module: a weighted and an
unweighted output must be distinguishable from their attributes alone.  Once
a file leaves the run there is nothing else to ask — two NetCDFs of the same
domain with the same variables are otherwise identical in shape, so an
operator holding one has no way to know how its mean was computed except by
reading what the run wrote down.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from teval import pipeline, workflow
from teval.config import IOConfig, StatsConfig, TevalConfig
from teval.weights.provenance import (
    APPLIED_ATTR,
    APPLIED_FALSE,
    APPLIED_TRUE,
    COVERAGE_ATTR,
    FILE_ATTR,
)

from tests.weighting_support import domain_map, weighted_config


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
        domain_map(raw_files),
        IOConfig(),
        weighted_config(weight_file, formulation_index_map),
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
        domain_map(raw_files),
        IOConfig(),
        weighted_config(partial_weight_file, formulation_index_map),
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
    results = workflow.load_domain_data(domain_map(raw_files), IOConfig(), StatsConfig())

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
        domain_map(raw_files),
        IOConfig(),
        weighted_config(weight_file, formulation_index_map),
    )
    unweighted = workflow.load_domain_data(
        domain_map(raw_files), IOConfig(), StatsConfig()
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
        weighted_config(weight_file, formulation_index_map)
        if weighted
        else StatsConfig()
    )
    config = TevalConfig(
        io=IOConfig(output_dir=tmp_path / "out", per_domain_output=False),
        stats=stats_config,
    )
    domain_dict = domain_map(raw_files)
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
        domain_map(raw_files, ensemble_file=ensemble_file),
        IOConfig(),
        weighted_config(weight_file, formulation_index_map),
    )

    attrs = _provenance_of(results)
    assert attrs[APPLIED_ATTR] == original[APPLIED_ATTR]
    assert attrs[FILE_ATTR] == original[FILE_ATTR]
    assert COVERAGE_ATTR not in attrs
