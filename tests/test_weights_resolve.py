"""
Tests for ``teval.weights.resolve``.

The resolver owns the *meaning* of a weight file: the legend binding, every
rule a weight group must obey, the expansion of per-nexus groups onto the
run's features, and the coverage policy.  The reader's job (schema, dtypes,
raising on unreadable input) is tested separately in
``test_weights_reader.py``; nothing here reads a file.

Two properties matter most and are asserted repeatedly:

* **Nothing silently wrong.** Every rule violation raises, and the message
  names the offending nexus, so a bad file is diagnosable without reading the
  source.  A rule that quietly dropped a member or handed a feature another
  nexus' weights would produce a plausible but wrong product.
* **Every returned row sums to 1.** Covered or uncovered, weighted or filled,
  the weights are a convex combination, so the weighted mean is a mean.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from teval.weights import (
    CoverageReport,
    bind_formulation_indices,
    resolve_weights,
    validate_weight_groups,
)
from teval.weights.resolve import ON_MISSING_POLICIES, SUM_TOLERANCE

RESOLVE_LOGGER = "teval.weights.resolve"


# --------------------------------------------------------------------- #
# Helpers and local fixtures                                            #
# --------------------------------------------------------------------- #
def tidy(rows) -> pd.DataFrame:
    """Build a tidy weight frame from ``(nexus_id, index, weight)`` triples."""
    return pd.DataFrame(
        list(rows), columns=["nexus_id", "formulation_index", "weight"]
    )


def group_rows(nexus_id, weights, start_index=1):
    """One complete group: one row per weight, indices numbered from 1."""
    return [
        (nexus_id, start_index + offset, weight)
        for offset, weight in enumerate(weights)
    ]


@pytest.fixture
def crosswalk():
    """
    Nexus to draining features, matching the ``flowpaths_frame`` fixture.

    Features 101, 102 and 103 converge on nexus 9001 — the confluence that
    makes the relationship genuinely many-to-one — and 201 drains to 9002.
    Built here rather than derived from the hydrofabric so the resolver is
    tested against the crosswalk *contract*, not against a second module.
    """
    return {9001: [101, 102, 103], 9002: [201]}


@pytest.fixture
def one_nexus_crosswalk():
    """Crosswalk covering only the confluence, leaving 201 uncovered."""
    return {9001: [101, 102, 103]}


# --------------------------------------------------------------------- #
# Legend binding — both directions of mismatch                          #
# --------------------------------------------------------------------- #
def test_binding_returns_indices_in_run_order(
    formulation_index_map, formulation_names
):
    """The result is positionally aligned with the run's formulation order."""
    assert bind_formulation_indices(formulation_index_map, formulation_names) == [
        1,
        2,
        3,
    ]


def test_binding_tracks_a_reordered_run(formulation_index_map):
    """A run that discovers formulations in another order rebinds, not re-maps."""
    assert bind_formulation_indices(
        formulation_index_map, ["formC", "formA", "formB"]
    ) == [3, 1, 2]


def test_binding_accepts_non_contiguous_indices():
    """Indices need only be unique and 1-based, not consecutive."""
    assert bind_formulation_indices({4: "a", 9: "b"}, ["b", "a"]) == [9, 4]


def test_map_naming_an_absent_formulation_raises(formulation_names):
    """A legend naming a formulation the run does not have is stale."""
    stale = {1: "formA", 2: "formB", 3: "formC", 4: "formD"}

    with pytest.raises(ValueError, match="not present in the run"):
        bind_formulation_indices(stale, formulation_names)


def test_formulation_missing_from_map_raises(formulation_names):
    """A discovered formulation no index names would silently drop out."""
    partial = {1: "formA", 2: "formB"}

    with pytest.raises(ValueError, match="missing from formulation_index_map"):
        bind_formulation_indices(partial, formulation_names)


def test_both_directions_of_mismatch_reported_together(formulation_names):
    """A legend wrong in both directions takes one run to diagnose, not two."""
    wrong = {1: "formA", 2: "formB", 3: "formZ"}

    with pytest.raises(ValueError) as excinfo:
        bind_formulation_indices(wrong, formulation_names)

    message = str(excinfo.value)
    assert "formZ" in message and "not present in the run" in message
    assert "formC" in message and "missing from formulation_index_map" in message


def test_empty_map_raises(formulation_names):
    """An empty legend cannot bind anything."""
    with pytest.raises(ValueError, match="formulation_index_map is empty"):
        bind_formulation_indices({}, formulation_names)


def test_empty_formulation_list_raises(formulation_index_map):
    """There must be something to bind against."""
    with pytest.raises(ValueError, match="at least one formulation"):
        bind_formulation_indices(formulation_index_map, [])


def test_map_naming_one_formulation_twice_raises(formulation_names):
    """Two indices for one formulation make the binding ambiguous."""
    with pytest.raises(ValueError, match="at most once"):
        bind_formulation_indices(
            {1: "formA", 2: "formA", 3: "formC"}, formulation_names
        )


def test_repeated_run_formulation_raises(formulation_index_map):
    """A repeated formulation in the run is rejected by the pure function too."""
    with pytest.raises(ValueError, match="must be unique"):
        bind_formulation_indices(
            formulation_index_map, ["formA", "formA", "formB", "formC"]
        )


def test_legend_mismatch_propagates_through_resolve(
    weight_frame, formulation_names, crosswalk, feature_ids
):
    """Binding runs first, so a stale legend fails before any other rule."""
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
    """A nexus present in the file must carry every configured index."""
    frame = tidy(
        group_rows("nex-9001", [0.5, 0.3, 0.2])
        + [("nex-9002", 1, 0.4), ("nex-9002", 2, 0.6)]
    )

    with pytest.raises(ValueError) as excinfo:
        validate_weight_groups(frame, formulation_index_map, formulation_names)

    message = str(excinfo.value)
    assert "incomplete weight group" in message
    assert "nex-9002" in message and "3" in message
    assert "nex-9001" not in message.split(".")[0]


def test_duplicate_row_raises(formulation_index_map, formulation_names):
    """Two rows for one (nexus, index) make the group ambiguous."""
    frame = tidy(group_rows("nex-9001", [0.5, 0.3, 0.2]) + [("nex-9001", 2, 0.3)])

    with pytest.raises(ValueError) as excinfo:
        validate_weight_groups(frame, formulation_index_map, formulation_names)

    message = str(excinfo.value)
    assert "duplicate rows" in message
    assert "nex-9001 index 2" in message


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


# --------------------------------------------------------------------- #
# Many-to-one expansion                                                 #
# --------------------------------------------------------------------- #
def test_expansion_returns_a_labelled_dense_array(
    weight_frame, formulation_index_map, formulation_names, crosswalk, feature_ids
):
    """Dims, coordinates and their order are part of the contract."""
    resolved, _ = resolve_weights(
        weight_frame,
        formulation_index_map,
        formulation_names,
        crosswalk,
        feature_ids,
    )

    assert isinstance(resolved, xr.DataArray)
    assert resolved.dims == ("feature_id", "formulation")
    assert resolved["feature_id"].values.tolist() == feature_ids
    assert resolved["formulation"].values.tolist() == formulation_names
    assert resolved.shape == (len(feature_ids), len(formulation_names))


def test_confluence_features_receive_identical_weights(
    weight_frame, formulation_index_map, formulation_names, crosswalk, feature_ids
):
    """101, 102 and 103 all drain to nexus 9001, so all three get its group."""
    resolved, _ = resolve_weights(
        weight_frame,
        formulation_index_map,
        formulation_names,
        crosswalk,
        feature_ids,
    )

    at_confluence = resolved.sel(feature_id=[101, 102, 103]).values
    assert np.array_equal(at_confluence[0], at_confluence[1])
    assert np.array_equal(at_confluence[0], at_confluence[2])
    assert at_confluence[0].tolist() == [0.5, 0.3, 0.2]


def test_each_nexus_group_lands_on_its_own_features(
    weight_frame, formulation_index_map, formulation_names, crosswalk, feature_ids
):
    """The other nexus' group is not broadcast over the confluence."""
    resolved, _ = resolve_weights(
        weight_frame,
        formulation_index_map,
        formulation_names,
        crosswalk,
        feature_ids,
    )

    assert resolved.sel(feature_id=201).values.tolist() == [0.25, 0.75, 0.0]


def test_every_row_sums_to_one(
    weight_frame, formulation_index_map, formulation_names, crosswalk, feature_ids
):
    """Covered or not, the weights are a convex combination."""
    resolved, _ = resolve_weights(
        weight_frame,
        formulation_index_map,
        formulation_names,
        crosswalk,
        feature_ids,
    )

    assert resolved.sum("formulation").values.tolist() == pytest.approx([1.0] * 4)


def test_expansion_follows_the_supplied_feature_order(
    weight_frame, formulation_index_map, formulation_names, crosswalk
):
    """The array is built for the dataset's coordinate, in its order."""
    resolved, _ = resolve_weights(
        weight_frame,
        formulation_index_map,
        formulation_names,
        crosswalk,
        [201, 103],
    )

    assert resolved["feature_id"].values.tolist() == [201, 103]
    assert resolved.values[0].tolist() == [0.25, 0.75, 0.0]
    assert resolved.values[1].tolist() == [0.5, 0.3, 0.2]


def test_expansion_follows_the_run_formulation_order(
    weight_frame, formulation_index_map, crosswalk, feature_ids
):
    """A reordered run reorders the columns, so index 1 still means formA."""
    resolved, _ = resolve_weights(
        weight_frame,
        formulation_index_map,
        ["formC", "formA", "formB"],
        crosswalk,
        feature_ids,
    )

    assert resolved.sel(feature_id=101).values.tolist() == [0.2, 0.5, 0.3]


def test_prefixed_and_integer_nexus_ids_match(
    formulation_index_map, formulation_names, feature_ids
):
    """``nex-9001`` in the file and ``9001`` in the crosswalk are one nexus."""
    frame = tidy(group_rows("nex-9001", [0.5, 0.3, 0.2]))

    resolved, report = resolve_weights(
        frame,
        formulation_index_map,
        formulation_names,
        {"nex-9001": [101, 102, 103]},
        feature_ids,
    )

    assert resolved.sel(feature_id=101).values.tolist() == [0.5, 0.3, 0.2]
    assert report.covered_features == 3


def test_float_spelled_nexus_ids_match_an_integer_crosswalk(
    formulation_index_map, formulation_names, feature_ids
):
    """
    ``9001.0`` is nexus 9001, not 90010.

    This is the spelling the reader produces from a weight file whose
    ``nexus_id`` column carries unprefixed ids and therefore lands as float
    dtype: ``astype(str)`` writes ``"9001.0"``.  Reducing it by dropping
    non-digits would swallow the decimal point and key a nexus that does not
    exist, and under the default ``on_missing`` policy the run would finish
    unweighted with nothing but a coverage warning.
    """
    frame = tidy(group_rows("9001.0", [0.5, 0.3, 0.2]))

    resolved, report = resolve_weights(
        frame,
        formulation_index_map,
        formulation_names,
        {9001: [101, 102, 103]},
        feature_ids,
    )

    assert resolved.sel(feature_id=101).values.tolist() == [0.5, 0.3, 0.2]
    assert report.covered_features == 3


def test_a_float_spelled_crosswalk_nexus_matches_the_weight_file(
    formulation_index_map, formulation_names, feature_ids
):
    """The same reduction runs on the crosswalk side, so it too reads 9001.0."""
    frame = tidy(group_rows("nex-9001", [0.5, 0.3, 0.2]))

    resolved, report = resolve_weights(
        frame,
        formulation_index_map,
        formulation_names,
        {9001.0: [101, 102, 103]},
        feature_ids,
    )

    assert resolved.sel(feature_id=101).values.tolist() == [0.5, 0.3, 0.2]
    assert report.covered_features == 3


def test_a_fractional_nexus_id_raises_rather_than_being_guessed_at(
    formulation_index_map, formulation_names, crosswalk, feature_ids
):
    """
    ``9001.5`` is not a nexus, and inventing one from its digits would hide that.

    Reducing it to 90015 would leave every feature on the equal-weight
    fallback, which reads as "the file did not cover this domain" rather than
    as "the file is wrong".
    """
    frame = tidy(group_rows("9001.5", [0.5, 0.3, 0.2]))

    with pytest.raises(ValueError, match="non-integer"):
        resolve_weights(
            frame,
            formulation_index_map,
            formulation_names,
            crosswalk,
            feature_ids,
        )


def test_colliding_nexus_ids_raise(
    formulation_index_map, formulation_names, crosswalk, feature_ids
):
    """Two spellings of one nexus number cannot be told apart after matching."""
    frame = tidy(
        group_rows("nex-9001", [0.5, 0.3, 0.2]) + group_rows("9001", [0.1, 0.2, 0.7])
    )

    with pytest.raises(ValueError, match="reduce to the same nexus"):
        resolve_weights(
            frame,
            formulation_index_map,
            formulation_names,
            crosswalk,
            feature_ids,
        )


def test_feature_under_two_nexuses_raises(
    weight_frame, formulation_index_map, formulation_names, feature_ids
):
    """A flowpath drains to exactly one nexus; two would be ambiguous weights."""
    bad_crosswalk = {9001: [101, 102, 103], 9002: [103, 201]}

    with pytest.raises(ValueError, match="more than one nexus"):
        resolve_weights(
            weight_frame,
            formulation_index_map,
            formulation_names,
            bad_crosswalk,
            feature_ids,
        )


def test_nexus_id_without_digits_raises(
    formulation_index_map, formulation_names, crosswalk, feature_ids
):
    """An id that reduces to nothing cannot be matched against the hydrofabric."""
    frame = tidy(group_rows("headwater", [0.5, 0.3, 0.2]))

    with pytest.raises(ValueError, match="carries no digits"):
        resolve_weights(
            frame,
            formulation_index_map,
            formulation_names,
            crosswalk,
            feature_ids,
        )


def test_boolean_crosswalk_key_raises(
    weight_frame, formulation_index_map, formulation_names, feature_ids
):
    """``True`` is an int in Python and would silently key nexus 1."""
    with pytest.raises(ValueError, match="not a nexus identifier"):
        resolve_weights(
            weight_frame,
            formulation_index_map,
            formulation_names,
            {True: [101]},
            feature_ids,
        )


def test_non_numeric_feature_ids_raise(
    weight_frame, formulation_index_map, formulation_names, crosswalk
):
    """Feature ids that are not numbers cannot address the dataset."""
    with pytest.raises(ValueError, match="must be integer feature ids"):
        resolve_weights(
            weight_frame,
            formulation_index_map,
            formulation_names,
            crosswalk,
            ["wb-101", "wb-201"],
        )


def test_non_integral_feature_ids_raise(
    weight_frame, formulation_index_map, formulation_names, crosswalk
):
    """Truncating 101.5 to 101 would weight the wrong flowpath."""
    with pytest.raises(ValueError, match="not integral"):
        resolve_weights(
            weight_frame,
            formulation_index_map,
            formulation_names,
            crosswalk,
            [101.5, 201.0],
        )


def test_duplicate_feature_ids_raise(
    weight_frame, formulation_index_map, formulation_names, crosswalk
):
    """The dataset's feature_id coordinate must be unique."""
    with pytest.raises(ValueError, match="duplicate value"):
        resolve_weights(
            weight_frame,
            formulation_index_map,
            formulation_names,
            crosswalk,
            [101, 101, 201],
        )


def test_empty_feature_ids_raise(
    weight_frame, formulation_index_map, formulation_names, crosswalk
):
    """Coverage is undefined for an empty run."""
    with pytest.raises(ValueError, match="No feature ids"):
        resolve_weights(
            weight_frame,
            formulation_index_map,
            formulation_names,
            crosswalk,
            [],
        )


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
