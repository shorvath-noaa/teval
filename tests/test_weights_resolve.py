"""
Tests for ``resolve_weights`` — the coverage policy and what it reports.

The public entry point, exercised where its two halves meet: features the
weight file does not reach fall back to equal weights or abort the run
according to ``on_missing``, and the ``CoverageReport`` says how much of the
run was actually weighted.  Validation is tested in
``test_weights_validate.py`` and expansion in ``test_weights_expand.py``.

Two properties matter most and are asserted repeatedly:

* **Every returned row sums to 1.** Covered or uncovered, weighted or filled,
  the weights are a convex combination, so the weighted mean is a mean.
* **The reported fraction is the fraction actually achieved**, counted over
  features rather than nexuses, so a confluence cannot inflate it.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from teval.weights import CoverageReport, resolve_weights
from teval.weights.resolve import ON_MISSING_POLICIES

from tests.weighting_support import RESOLVE_LOGGER, group_rows, tidy


# --------------------------------------------------------------------- #
# Coverage policy — warn falls back, error aborts                       #
# --------------------------------------------------------------------- #
def test_uncovered_features_fall_back_to_equal_weights(
    formulation_index_map, formulation_names, crosswalk, feature_ids, caplog
):
    """An uncovered feature behaves exactly as it did before weighting."""
    frame = tidy(group_rows("nex-9001", [0.5, 0.3, 0.2]))

    with caplog.at_level(logging.WARNING, logger=RESOLVE_LOGGER):
        resolved, report = resolve_weights(
            frame,
            formulation_index_map,
            formulation_names,
            crosswalk,
            feature_ids,
            on_missing="warn",
        )

    assert resolved.sel(feature_id=101).values.tolist() == [0.5, 0.3, 0.2]
    assert resolved.sel(feature_id=201).values.tolist() == pytest.approx(
        [1 / 3, 1 / 3, 1 / 3]
    )
    assert report.uncovered_features == 1


def test_warn_logs_the_counts_and_the_fraction(
    formulation_index_map, formulation_names, crosswalk, feature_ids, caplog
):
    """The warning has to say how much of the domain was actually weighted."""
    frame = tidy(group_rows("nex-9001", [0.5, 0.3, 0.2]))

    with caplog.at_level(logging.WARNING, logger=RESOLVE_LOGGER):
        resolve_weights(
            frame,
            formulation_index_map,
            formulation_names,
            crosswalk,
            feature_ids,
            on_missing="warn",
        )

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    text = warnings[0].getMessage()
    assert "75.0%" in text
    assert "3 of 4" in text
    assert "1 uncovered" in text
    assert "equal weights" in text


def test_no_warning_at_full_coverage(
    weight_frame, formulation_index_map, formulation_names, crosswalk, feature_ids, caplog
):
    """A fully covered run must not train the user to ignore the warning."""
    with caplog.at_level(logging.WARNING, logger=RESOLVE_LOGGER):
        _, report = resolve_weights(
            weight_frame,
            formulation_index_map,
            formulation_names,
            crosswalk,
            feature_ids,
            on_missing="warn",
        )

    assert report.is_complete
    assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []


def test_error_policy_aborts_on_incomplete_coverage(
    formulation_index_map, formulation_names, crosswalk, feature_ids
):
    """Under 'error' a partial file is a hard failure, not a quiet fallback."""
    frame = tidy(group_rows("nex-9001", [0.5, 0.3, 0.2]))

    with pytest.raises(ValueError) as excinfo:
        resolve_weights(
            frame,
            formulation_index_map,
            formulation_names,
            crosswalk,
            feature_ids,
            on_missing="error",
        )

    message = str(excinfo.value)
    assert "on_missing" in message
    assert "75.0%" in message


def test_error_policy_passes_at_full_coverage(
    weight_frame, formulation_index_map, formulation_names, crosswalk, feature_ids
):
    """'error' only fires on a gap — full coverage is not an error."""
    _, report = resolve_weights(
        weight_frame,
        formulation_index_map,
        formulation_names,
        crosswalk,
        feature_ids,
        on_missing="error",
    )

    assert report.is_complete


def test_unknown_on_missing_policy_raises(
    weight_frame, formulation_index_map, formulation_names, crosswalk, feature_ids
):
    """A typo in the policy must not be read as the permissive branch."""
    with pytest.raises(ValueError, match="Unknown on_missing policy"):
        resolve_weights(
            weight_frame,
            formulation_index_map,
            formulation_names,
            crosswalk,
            feature_ids,
            on_missing="ignore",
        )
    assert set(ON_MISSING_POLICIES) == {"warn", "error"}


def test_feature_absent_from_the_crosswalk_is_uncovered(
    weight_frame,
    formulation_index_map,
    formulation_names,
    one_nexus_crosswalk,
    feature_ids,
):
    """A feature the crosswalk does not place falls through to equal weights."""
    resolved, report = resolve_weights(
        weight_frame,
        formulation_index_map,
        formulation_names,
        one_nexus_crosswalk,
        feature_ids,
    )

    assert resolved.sel(feature_id=201).values.tolist() == pytest.approx(
        [1 / 3, 1 / 3, 1 / 3]
    )
    assert report.uncovered_features == 1


def test_empty_crosswalk_leaves_everything_uncovered(
    weight_frame, formulation_index_map, formulation_names, feature_ids
):
    """No hydrofabric means no join, and every feature keeps the simple mean."""
    resolved, report = resolve_weights(
        weight_frame,
        formulation_index_map,
        formulation_names,
        {},
        feature_ids,
    )

    assert report.covered_features == 0
    assert report.fraction == 0.0
    assert np.allclose(resolved.values, 1 / 3)


def test_empty_weight_frame_leaves_everything_uncovered(
    formulation_index_map, formulation_names, crosswalk, feature_ids
):
    """An empty file is a coverage outcome, and equal weights are the result."""
    empty = tidy([]).astype(
        {"nexus_id": str, "formulation_index": int, "weight": float}
    )

    resolved, report = resolve_weights(
        empty, formulation_index_map, formulation_names, crosswalk, feature_ids
    )

    assert report.covered_features == 0
    assert np.allclose(resolved.values, 1 / 3)


# --------------------------------------------------------------------- #
# The coverage report                                                   #
# --------------------------------------------------------------------- #
def test_reported_fraction_is_accurate_at_full_coverage(
    weight_frame, formulation_index_map, formulation_names, crosswalk, feature_ids
):
    """Four of four features weighted is 100%."""
    _, report = resolve_weights(
        weight_frame,
        formulation_index_map,
        formulation_names,
        crosswalk,
        feature_ids,
    )

    assert report == CoverageReport(
        total_features=4,
        covered_features=4,
        uncovered_features=0,
        fraction=1.0,
        used_nexus=2,
        unused_nexus=0,
    )


def test_reported_fraction_counts_features_not_nexuses(
    formulation_index_map, formulation_names, crosswalk, feature_ids
):
    """
    One nexus of two is covered, but it carries three of four features.

    Counting nexuses would report 50%; the number the user needs is 75%,
    because that is the share of the product that is actually weighted.
    """
    frame = tidy(group_rows("nex-9001", [0.5, 0.3, 0.2]))

    _, report = resolve_weights(
        frame,
        formulation_index_map,
        formulation_names,
        crosswalk,
        feature_ids,
    )

    assert report.fraction == pytest.approx(0.75)
    assert (report.covered_features, report.uncovered_features) == (3, 1)
    assert report.used_nexus == 1
    assert not report.is_complete


def test_unused_groups_are_reported_but_not_an_error(
    weight_frame, formulation_index_map, formulation_names, feature_ids
):
    """A national file applied to one domain leaves most groups unused."""
    extra = pd.concat(
        [weight_frame, tidy(group_rows("nex-9999", [0.4, 0.4, 0.2]))],
        ignore_index=True,
    )

    _, report = resolve_weights(
        extra,
        formulation_index_map,
        formulation_names,
        {9001: [101, 102, 103], 9002: [201]},
        feature_ids,
        on_missing="error",
    )

    assert report.used_nexus == 2
    assert report.unused_nexus == 1
    assert report.is_complete


def test_report_summary_states_the_coverage(
    formulation_index_map, formulation_names, crosswalk, feature_ids
):
    """``summary`` is reused by the log line, the error and NetCDF provenance."""
    frame = tidy(group_rows("nex-9001", [0.5, 0.3, 0.2]))

    _, report = resolve_weights(
        frame,
        formulation_index_map,
        formulation_names,
        crosswalk,
        feature_ids,
    )

    summary = report.summary()
    assert "75.0%" in summary
    assert "3 of 4" in summary
    assert "1 uncovered" in summary


# --------------------------------------------------------------------- #
# The resolved array is usable downstream                               #
# --------------------------------------------------------------------- #
def test_resolved_weights_multiply_a_lazy_dataset_lazily(
    weight_frame,
    formulation_index_map,
    formulation_names,
    crosswalk,
    feature_ids,
    combined_ds,
):
    """
    The whole point: a weighted sum over ``formulation``, still lazy.

    Asserted against a hand-computed value so the alignment by coordinate
    label is checked, not just the shape.  At t=0, feature 101 the members
    are 110, 210 and 310, weighted 0.5/0.3/0.2 → 55 + 63 + 62 = 180.
    """
    resolved, _ = resolve_weights(
        weight_frame,
        formulation_index_map,
        formulation_names,
        crosswalk,
        feature_ids,
    )

    weighted = (combined_ds["streamflow"] * resolved).sum("formulation")

    assert hasattr(weighted.data, "dask")
    assert weighted.isel(time=0).sel(feature_id=101).compute().item() == pytest.approx(
        180.0
    )


def test_alignment_is_by_label_not_position(
    weight_frame,
    formulation_index_map,
    formulation_names,
    crosswalk,
    feature_ids,
    combined_ds,
):
    """A weight array built in another formulation order still gives the same mean."""
    reversed_names = list(reversed(formulation_names))
    straight, _ = resolve_weights(
        weight_frame,
        formulation_index_map,
        formulation_names,
        crosswalk,
        feature_ids,
    )
    flipped, _ = resolve_weights(
        weight_frame,
        formulation_index_map,
        reversed_names,
        crosswalk,
        feature_ids,
    )

    from_straight = (combined_ds["streamflow"] * straight).sum("formulation")
    from_flipped = (combined_ds["streamflow"] * flipped).sum("formulation")

    xr.testing.assert_allclose(from_straight.compute(), from_flipped.compute())
