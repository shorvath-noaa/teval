"""
Tests for the many-to-one expansion of weight groups onto features.

The resolver's second half: a weight group is stated per *nexus*, but the
statistics are computed per *feature*, and several features can drain to one
nexus.  This module covers that expansion — every draining feature receiving
its nexus' group, the ordering of both axes of the returned array, and the
nexus identifier matching that decides which group a feature is handed.

The identifier tests carry the most weight: a mismatch there does not raise,
it silently hands a feature equal weights or another nexus' group, so the
cases that must *match* (prefixed against integer, float-spelled against
integer) and the cases that must *raise* (fractional, colliding, digitless)
are asserted separately.  The rules a group must obey before it is expanded
are in ``test_weights_validate.py``; the coverage policy applied afterwards is
in ``test_weights_resolve.py``.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from teval.weights import resolve_weights

from tests.weighting_support import group_rows, tidy


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
