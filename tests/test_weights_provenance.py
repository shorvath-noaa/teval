"""
Tests for the provenance attributes recorded alongside the ensemble statistics.

The attributes exist to answer one question from the output file alone: is
this mean weighted?  So the tests here care about two things — that a weighted
and an unweighted run are told apart by an attribute that is always present,
and that the values chosen survive the trip through NetCDF, which has a
narrower type vocabulary than Python and would happily reject a ``bool``.

Wiring these into a real run is covered in ``test_workflow_weights.py``; what
is pinned here is the mapping itself.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from teval.config import WeightsConfig
from teval.weights import CoverageReport, weighting_attrs
from teval.weights.provenance import (
    APPLIED_ATTR,
    APPLIED_FALSE,
    APPLIED_TRUE,
    COVERAGE_ATTR,
    FILE_ATTR,
)


def _report(covered=4, total=4, uncovered=None):
    """A coverage report with a consistent fraction, for the shape of it."""
    uncovered = total - covered if uncovered is None else uncovered
    return CoverageReport(
        total_features=total,
        covered_features=covered,
        uncovered_features=uncovered,
        fraction=covered / total,
        used_nexus=2,
        unused_nexus=0,
    )


def _config(path="weights.csv", **overrides):
    return WeightsConfig(
        file=Path(path),
        formulation_index_map={1: "cfe", 2: "noahowp"},
        **overrides,
    )


# --------------------------------------------------------------------- #
# What each kind of run records                                         #
# --------------------------------------------------------------------- #
def test_a_weighted_run_records_the_file_and_the_coverage():
    """All three attributes, with the file named as the configuration named it."""
    attrs = weighting_attrs(_config("/data/national_weights.parquet"), _report())

    assert attrs == {
        APPLIED_ATTR: APPLIED_TRUE,
        FILE_ATTR: "/data/national_weights.parquet",
        COVERAGE_ATTR: 1.0,
    }


def test_an_unweighted_run_records_that_weighting_was_not_applied():
    """
    The flag is written either way; that is the whole point of it.

    An unweighted run that recorded nothing would be indistinguishable from
    one whose weighting attributes were lost, and a reader would be left
    inferring from the values — which is exactly what these attributes are
    for.
    """
    assert weighting_attrs() == {APPLIED_ATTR: APPLIED_FALSE}


def test_the_two_kinds_of_run_disagree_on_the_flag():
    """Stated directly, since 'distinguishable' is the requirement."""
    weighted = weighting_attrs(_config(), _report())
    unweighted = weighting_attrs()

    assert weighted[APPLIED_ATTR] != unweighted[APPLIED_ATTR]


def test_an_unweighted_run_records_no_file_and_no_coverage():
    """
    Absent rather than zero-filled.

    A coverage of 0.0 means weighting was attempted and reached nothing, which
    is a real and much worse outcome than not having asked for weights; the
    two must not be written the same way.
    """
    attrs = weighting_attrs()

    assert FILE_ATTR not in attrs
    assert COVERAGE_ATTR not in attrs


def test_partial_coverage_is_recorded_as_the_achieved_fraction():
    """The fraction reported is the resolution's, not a rounded stand-in."""
    attrs = weighting_attrs(_config(), _report(covered=3, total=4))

    assert attrs[COVERAGE_ATTR] == pytest.approx(0.75)


def test_zero_coverage_still_records_weighting_as_applied():
    """
    Weights were configured and resolved, so the run *was* the weighted path.

    Every feature fell back to equal weights, which makes the values identical
    to an unweighted run — so the coverage fraction is the only thing that
    says so, and the flag must not quietly flip to 'false' and hide it.
    """
    attrs = weighting_attrs(_config(), _report(covered=0, total=4))

    assert attrs[APPLIED_ATTR] == APPLIED_TRUE
    assert attrs[COVERAGE_ATTR] == pytest.approx(0.0)


# --------------------------------------------------------------------- #
# Misuse                                                                #
# --------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "config, report",
    [
        pytest.param(_config(), None, id="config-without-report"),
        pytest.param(None, _report(), id="report-without-config"),
    ],
)
def test_half_a_weighted_run_is_rejected(config, report):
    """
    The pair travels together; one alone means the caller lost track.

    Filling in the other half by guessing would write provenance that
    misdescribes the file, which is worse than the run failing here.
    """
    with pytest.raises(ValueError, match="both"):
        weighting_attrs(config, report)


# --------------------------------------------------------------------- #
# The values have to survive NetCDF                                     #
# --------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "attrs",
    [
        pytest.param(weighting_attrs(_config("w.csv"), _report(3, 4)), id="weighted"),
        pytest.param(weighting_attrs(), id="unweighted"),
    ],
)
def test_the_attributes_survive_a_netcdf_round_trip(tmp_path, attrs):
    """
    NetCDF has no boolean attribute type, and h5netcdf refuses one outright.

    Everything above would pass with ``True``/``False`` in place of the
    strings, and the run would then fail at the write — after the compute.
    This writes with the same engine the pipeline uses and reads the values
    back to confirm they arrive intact.
    """
    path = tmp_path / "ensemble.nc"
    xr.Dataset(
        {"streamflow_mean": ("feature_id", np.array([1.0, 2.0]))},
        coords={"feature_id": [101, 102]},
        attrs=attrs,
    ).to_netcdf(path, engine="h5netcdf")

    with xr.open_dataset(path, engine="h5netcdf") as reopened:
        for key, value in attrs.items():
            assert reopened.attrs[key] == value
