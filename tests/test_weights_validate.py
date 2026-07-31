"""
Tests for ``validate_weight_groups`` — every rule a weight file must obey.

This is the resolver's first half: the legend that binds a
``formulation_index`` to a formulation name, and the rules the resulting
groups must satisfy before anything is expanded onto features.  Expansion and
the coverage policy are the second half, tested in ``test_weights_expand.py``
and ``test_weights_resolve.py``; the reader's job (schema, dtypes, raising on
unreadable input) is tested in ``test_weights_reader.py``.  Nothing here reads
a file.

The property asserted throughout is that **nothing is silently wrong**: every
violation raises, and the message names the offending nexus, so a bad file is
diagnosable without reading the source.  A rule that quietly dropped a member
or renumbered a group would produce a plausible but wrong product.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from teval.weights import resolve_weights, validate_weight_groups
from teval.weights.resolve import SUM_TOLERANCE

from tests.weighting_support import group_rows, tidy


# --------------------------------------------------------------------- #
# The legend — one set comparison, both directions of mismatch          #
# --------------------------------------------------------------------- #
# The index is a file-format detail: it is relabelled to a formulation name
# once, at the top of validate_weight_groups, and is never an output.  So the
# legend is asserted through the public boundary, on the groups it produces
# and the errors it raises, rather than on a binding function.
def test_non_contiguous_indices_are_accepted():
    """Indices need only be defined by the legend, not consecutive."""
    frame = tidy([("nex-9001", 4, 0.6), ("nex-9001", 9, 0.4)])

    groups = validate_weight_groups(
        frame, {4: "formA", 9: "formB"}, ["formB", "formA"]
    )

    assert list(groups.columns) == ["formB", "formA"]
    assert groups.loc["nex-9001"].tolist() == [0.4, 0.6]


def test_map_naming_an_absent_formulation_raises(weight_frame, formulation_names):
    """A legend naming a formulation the run does not have is stale."""
    stale = {1: "formA", 2: "formB", 3: "formC", 4: "formD"}

    with pytest.raises(ValueError, match="not present in the run"):
        validate_weight_groups(weight_frame, stale, formulation_names)


def test_formulation_missing_from_map_raises(weight_frame, formulation_names):
    """A discovered formulation no index names would silently drop out."""
    partial = {1: "formA", 2: "formB"}

    with pytest.raises(ValueError, match="missing from formulation_index_map"):
        validate_weight_groups(weight_frame, partial, formulation_names)


def test_both_directions_of_mismatch_reported_together(
    weight_frame, formulation_names
):
    """A legend wrong in both directions takes one run to diagnose, not two."""
    wrong = {1: "formA", 2: "formB", 3: "formZ"}

    with pytest.raises(ValueError) as excinfo:
        validate_weight_groups(weight_frame, wrong, formulation_names)

    message = str(excinfo.value)
    assert "formZ" in message and "not present in the run" in message
    assert "formC" in message and "missing from formulation_index_map" in message


def test_empty_map_raises(weight_frame, formulation_names):
    """An empty legend names none of the run's formulations."""
    with pytest.raises(ValueError, match="missing from formulation_index_map"):
        validate_weight_groups(weight_frame, {}, formulation_names)


def test_empty_formulation_list_raises(weight_frame, formulation_index_map):
    """There must be something to weight."""
    with pytest.raises(ValueError, match="at least one formulation"):
        validate_weight_groups(weight_frame, formulation_index_map, [])


def test_map_naming_one_formulation_twice_raises(weight_frame, formulation_names):
    """Two indices spent on one name leave another formulation unnamed."""
    with pytest.raises(ValueError, match="missing from formulation_index_map") as exc:
        validate_weight_groups(
            weight_frame, {1: "formA", 2: "formA", 3: "formC"}, formulation_names
        )

    assert "formB" in str(exc.value)


def test_repeated_run_formulation_raises(weight_frame, formulation_index_map):
    """A repeated formulation in the run makes the run's own order ambiguous."""
    with pytest.raises(ValueError, match="must be unique"):
        validate_weight_groups(
            weight_frame, formulation_index_map, ["formA", "formA", "formB", "formC"]
        )


def test_the_legend_is_checked_before_the_frame(formulation_names):
    """A stale legend is reported even when the frame is unreadable too."""
    malformed = tidy([("nex-9001", 1, 0.5)]).drop(columns=["weight"])

    with pytest.raises(ValueError, match="missing from formulation_index_map"):
        validate_weight_groups(malformed, {1: "formA"}, formulation_names)


def test_legend_mismatch_propagates_through_resolve(
    weight_frame, formulation_names, crosswalk, feature_ids
):
    """The legend is checked first, so a stale one fails before any other rule."""
    with pytest.raises(ValueError, match="missing from formulation_index_map"):
        resolve_weights(
            weight_frame,
            {1: "formA", 2: "formB"},
            formulation_names,
            crosswalk,
            feature_ids,
        )


# --------------------------------------------------------------------- #
# The validated happy path                                              #
# --------------------------------------------------------------------- #
def test_validate_places_every_weight_in_the_right_cell(
    weight_frame, formulation_index_map, formulation_names
):
    """The pivot is asserted cell by cell against the fixture's values."""
    groups = validate_weight_groups(
        weight_frame, formulation_index_map, formulation_names
    )

    assert list(groups.index) == ["nex-9001", "nex-9002"]
    assert list(groups.columns) == formulation_names
    assert groups.loc["nex-9001"].tolist() == [0.5, 0.3, 0.2]
    assert groups.loc["nex-9002"].tolist() == [0.25, 0.75, 0.0]


def test_validated_columns_follow_run_order_not_file_order(
    weight_frame, formulation_index_map
):
    """Columns track the dataset's formulation order, so index 1 stays formA."""
    groups = validate_weight_groups(
        weight_frame, formulation_index_map, ["formC", "formA", "formB"]
    )

    assert list(groups.columns) == ["formC", "formA", "formB"]
    assert groups.loc["nex-9001"].tolist() == [0.2, 0.5, 0.3]


def test_file_row_order_does_not_matter(
    weight_frame, formulation_index_map, formulation_names
):
    """A shuffled file validates to the same groups."""
    shuffled = weight_frame.iloc[[4, 0, 5, 2, 3, 1]].reset_index(drop=True)

    expected = validate_weight_groups(
        weight_frame, formulation_index_map, formulation_names
    )
    result = validate_weight_groups(
        shuffled, formulation_index_map, formulation_names
    )

    assert result.loc["nex-9001"].tolist() == expected.loc["nex-9001"].tolist()
    assert result.loc["nex-9002"].tolist() == expected.loc["nex-9002"].tolist()


def test_validation_does_not_mutate_the_input_frame(
    weight_frame, formulation_index_map, formulation_names
):
    """The resolver is pure — the caller's frame comes back untouched."""
    before = weight_frame.copy(deep=True)

    validate_weight_groups(weight_frame, formulation_index_map, formulation_names)

    pd.testing.assert_frame_equal(weight_frame, before)


def test_empty_frame_validates_to_empty_groups(
    formulation_index_map, formulation_names
):
    """No rows is a coverage outcome, not a validation failure."""
    empty = tidy([]).astype(
        {"nexus_id": str, "formulation_index": int, "weight": float}
    )

    groups = validate_weight_groups(empty, formulation_index_map, formulation_names)

    assert groups.empty
    assert list(groups.columns) == formulation_names


def test_missing_schema_column_raises(formulation_index_map, formulation_names):
    """The resolver enforces the schema rather than trusting its caller."""
    frame = tidy(group_rows("nex-9001", [0.5, 0.3, 0.2])).drop(columns=["weight"])

    with pytest.raises(ValueError, match="missing required column"):
        validate_weight_groups(frame, formulation_index_map, formulation_names)


def test_uncoercible_column_raises(formulation_index_map, formulation_names):
    """A frame built by hand, bypassing the reader, is still type-checked."""
    frame = tidy([("nex-9001", 1, "half"), ("nex-9001", 2, 0.3), ("nex-9001", 3, 0.2)])

    with pytest.raises(ValueError, match="could not be read as"):
        validate_weight_groups(frame, formulation_index_map, formulation_names)


# --------------------------------------------------------------------- #
# Completeness — a missing row and a duplicate row each raise            #
# --------------------------------------------------------------------- #
def test_missing_row_raises(formulation_index_map, formulation_names):
    """A nexus present in the file must carry every formulation in the run."""
    frame = tidy(
        group_rows("nex-9001", [0.5, 0.3, 0.2])
        + [("nex-9002", 1, 0.4), ("nex-9002", 2, 0.6)]
    )

    with pytest.raises(ValueError) as excinfo:
        validate_weight_groups(frame, formulation_index_map, formulation_names)

    message = str(excinfo.value)
    assert "incomplete weight group" in message
    # The member that dropped out is named, not the index the file spelled it
    # with: the reader of the message has the formulation names, not the legend.
    assert "nex-9002" in message and "formC" in message
    assert "nex-9001" not in message.split(".")[0]


def test_duplicate_row_raises(formulation_index_map, formulation_names):
    """Two rows for one (nexus, index) make the group ambiguous."""
    frame = tidy(group_rows("nex-9001", [0.5, 0.3, 0.2]) + [("nex-9001", 2, 0.3)])

    with pytest.raises(ValueError) as excinfo:
        validate_weight_groups(frame, formulation_index_map, formulation_names)

    message = str(excinfo.value)
    assert "duplicate rows" in message
    assert "nex-9001 formulation formB" in message


def test_duplicate_row_raises_even_when_values_agree(
    formulation_index_map, formulation_names
):
    """A repeat that happens to agree is still the concatenated-file signature."""
    frame = tidy(
        group_rows("nex-9001", [0.5, 0.3, 0.2])
        + group_rows("nex-9001", [0.5, 0.3, 0.2])
    )

    with pytest.raises(ValueError, match="duplicate rows"):
        validate_weight_groups(frame, formulation_index_map, formulation_names)


def test_unknown_index_raises(formulation_index_map, formulation_names):
    """An index the legend does not define would drop a supplied weight."""
    frame = tidy(group_rows("nex-9001", [0.5, 0.3, 0.1]) + [("nex-9001", 7, 0.1)])

    with pytest.raises(ValueError) as excinfo:
        validate_weight_groups(frame, formulation_index_map, formulation_names)

    message = str(excinfo.value)
    assert "formulation_index_map does not" in message
    assert "7" in message and "nex-9001" in message


def test_a_nexus_with_no_rows_is_not_a_completeness_error(
    formulation_index_map, formulation_names
):
    """Absence is coverage, governed by on_missing — not a validation failure."""
    frame = tidy(group_rows("nex-9001", [0.5, 0.3, 0.2]))

    groups = validate_weight_groups(frame, formulation_index_map, formulation_names)

    assert list(groups.index) == ["nex-9001"]


# --------------------------------------------------------------------- #
# Sign, zeros and all-zero groups                                       #
# --------------------------------------------------------------------- #
def test_negative_weight_raises(formulation_index_map, formulation_names):
    """A sign error upstream must not produce a meaningless combination."""
    frame = tidy(group_rows("nex-9001", [1.3, -0.3, 0.0]))

    with pytest.raises(ValueError) as excinfo:
        validate_weight_groups(frame, formulation_index_map, formulation_names)

    message = str(excinfo.value)
    assert "negative weights" in message and "nex-9001" in message


def test_negative_weight_raises_under_normalize(
    formulation_index_map, formulation_names
):
    """Normalizing must not launder a negative into a plausible-looking group."""
    frame = tidy(group_rows("nex-9001", [1.3, -0.3, 0.0]))

    with pytest.raises(ValueError, match="negative weights"):
        validate_weight_groups(
            frame, formulation_index_map, formulation_names, normalize=True
        )


def test_non_finite_weight_raises(formulation_index_map, formulation_names):
    """A NaN would poison every value it touched."""
    frame = tidy(group_rows("nex-9001", [0.5, np.nan, 0.5]))

    with pytest.raises(ValueError, match="non-finite weights"):
        validate_weight_groups(frame, formulation_index_map, formulation_names)


def test_all_zero_group_raises(formulation_index_map, formulation_names):
    """A group of zeros would silently produce zero flow at that location."""
    frame = tidy(group_rows("nex-9001", [0.0, 0.0, 0.0]))

    with pytest.raises(ValueError) as excinfo:
        validate_weight_groups(frame, formulation_index_map, formulation_names)

    message = str(excinfo.value)
    assert "all-zero weight group" in message and "nex-9001" in message
    assert "Individual zero weights are permitted" in message


def test_all_zero_group_raises_under_normalize(
    formulation_index_map, formulation_names
):
    """Dividing by zero must be reported as the all-zero group it is."""
    frame = tidy(group_rows("nex-9001", [0.0, 0.0, 0.0]))

    with pytest.raises(ValueError, match="all-zero weight group"):
        validate_weight_groups(
            frame, formulation_index_map, formulation_names, normalize=True
        )


def test_individual_zero_is_permitted(
    weight_frame, formulation_index_map, formulation_names
):
    """One zero excludes one formulation at one location, deliberately."""
    groups = validate_weight_groups(
        weight_frame, formulation_index_map, formulation_names
    )

    assert groups.loc["nex-9002", "formC"] == 0.0
    assert groups.loc["nex-9002"].sum() == pytest.approx(1.0)


# --------------------------------------------------------------------- #
# Sum to one, and the tolerance either side                             #
# --------------------------------------------------------------------- #
def test_group_summing_to_one_is_accepted(formulation_index_map, formulation_names):
    """The straightforward case, including one that is not exact in binary."""
    frame = tidy(group_rows("nex-9001", [0.1, 0.2, 0.7]))

    groups = validate_weight_groups(frame, formulation_index_map, formulation_names)

    assert groups.loc["nex-9001"].sum() == pytest.approx(1.0)


def test_sum_just_inside_tolerance_is_accepted(
    formulation_index_map, formulation_names
):
    """Float representation error in a legitimate file must not be rejected."""
    frame = tidy(group_rows("nex-9001", [0.5, 0.3, 0.2 + 9e-7]))

    groups = validate_weight_groups(frame, formulation_index_map, formulation_names)

    assert abs(groups.loc["nex-9001"].sum() - 1.0) < SUM_TOLERANCE


def test_sum_just_outside_tolerance_raises(
    formulation_index_map, formulation_names
):
    """A real error just past the boundary is caught."""
    frame = tidy(group_rows("nex-9001", [0.5, 0.3, 0.2 + 1.1e-6]))

    with pytest.raises(ValueError) as excinfo:
        validate_weight_groups(frame, formulation_index_map, formulation_names)

    message = str(excinfo.value)
    assert "do not sum to 1" in message
    assert "nex-9001" in message
    assert "normalize" in message


def test_sum_error_names_only_the_offending_group(
    formulation_index_map, formulation_names
):
    """A good group alongside a bad one is not reported as at fault."""
    frame = tidy(
        group_rows("nex-9001", [0.5, 0.3, 0.2]) + group_rows("nex-9002", [0.5, 0.3, 0.3])
    )

    with pytest.raises(ValueError) as excinfo:
        validate_weight_groups(frame, formulation_index_map, formulation_names)

    message = str(excinfo.value)
    assert "nex-9002" in message
    assert "nex-9001" not in message


def test_tolerance_is_configurable(formulation_index_map, formulation_names):
    """A caller may tighten the rule; the default is only a default."""
    frame = tidy(group_rows("nex-9001", [0.5, 0.3, 0.2 + 5e-7]))

    with pytest.raises(ValueError, match="do not sum to 1"):
        validate_weight_groups(
            frame, formulation_index_map, formulation_names, tolerance=1e-9
        )


# --------------------------------------------------------------------- #
# normalize accepts an arbitrary positive scale                         #
# --------------------------------------------------------------------- #
@pytest.mark.parametrize("scale", [10.0, 400.0, 1e-9, 3.7])
def test_normalize_accepts_any_positive_scale(
    scale, formulation_index_map, formulation_names
):
    """Counts, percentages or tiny magnitudes all normalize to a valid group."""
    frame = tidy(group_rows("nex-9001", [0.5 * scale, 0.3 * scale, 0.2 * scale]))

    groups = validate_weight_groups(
        frame, formulation_index_map, formulation_names, normalize=True
    )

    assert groups.loc["nex-9001"].sum() == pytest.approx(1.0)
    assert groups.loc["nex-9001"].tolist() == pytest.approx([0.5, 0.3, 0.2])


def test_normalize_scales_each_group_independently(
    formulation_index_map, formulation_names
):
    """Groups are normalized by their own sum, not by a shared total."""
    frame = tidy(
        group_rows("nex-9001", [2.0, 1.0, 1.0]) + group_rows("nex-9002", [30.0, 10.0, 10.0])
    )

    groups = validate_weight_groups(
        frame, formulation_index_map, formulation_names, normalize=True
    )

    assert groups.loc["nex-9001"].tolist() == pytest.approx([0.5, 0.25, 0.25])
    assert groups.loc["nex-9002"].tolist() == pytest.approx([0.6, 0.2, 0.2])


def test_normalize_bypasses_the_sum_rule(formulation_index_map, formulation_names):
    """A file that would be rejected by default is accepted with normalize."""
    frame = tidy(group_rows("nex-9001", [1.0, 1.0, 1.0]))

    with pytest.raises(ValueError, match="do not sum to 1"):
        validate_weight_groups(frame, formulation_index_map, formulation_names)

    groups = validate_weight_groups(
        frame, formulation_index_map, formulation_names, normalize=True
    )
    assert groups.loc["nex-9001"].tolist() == pytest.approx([1 / 3, 1 / 3, 1 / 3])


def test_normalize_preserves_an_individual_zero(
    formulation_index_map, formulation_names
):
    """A deliberate exclusion survives normalization."""
    frame = tidy(group_rows("nex-9001", [3.0, 1.0, 0.0]))

    groups = validate_weight_groups(
        frame, formulation_index_map, formulation_names, normalize=True
    )

    assert groups.loc["nex-9001"].tolist() == pytest.approx([0.75, 0.25, 0.0])
