"""
Tests for the nexus-to-feature crosswalk built from the hydrofabric.

The crosswalk is the join between weights, which are supplied per nexus, and
the ensemble dataset, which is indexed by feature id.  Its central hazard is
that ``load_hydrofabric`` strips the ``wb-`` and ``nex-`` prefixes, after which
a nexus number and a flowpath id are indistinguishable by value — so a builder
that read the wrong column would return plausible, silently wrong weights.
These tests therefore assert not only the mapping but *which column* it came
from.
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest

from teval.io import build_nexus_crosswalk
from teval.io.hydrofabric import build_nexus_crosswalk as build_from_module
from teval.weights.resolve import resolve_weights


# --------------------------------------------------------------------- #
# The shape the plan calls for: a confluence                            #
# --------------------------------------------------------------------- #
def test_confluence_gives_every_flowpath_the_same_nexus(flowpaths_frame):
    """Several flowpaths sharing one toid land in one group; the relationship
    is many-to-one and the builder must not collapse it to one feature."""
    assert build_nexus_crosswalk(flowpaths_frame) == {
        9001: [101, 102, 103],
        9002: [201],
    }


def test_exported_from_the_io_package(flowpaths_frame):
    """The package export and the module attribute are the same function."""
    assert build_nexus_crosswalk is build_from_module


def test_keys_and_values_are_plain_ints(flowpaths_frame):
    """numpy scalars would key a dict by identity-compatible but surprising
    types; the crosswalk is a plain-Python structure."""
    crosswalk = build_nexus_crosswalk(flowpaths_frame)

    for nexus_id, features in crosswalk.items():
        assert type(nexus_id) is int
        assert all(type(feature_id) is int for feature_id in features)


def test_order_is_deterministic_and_follows_the_frame():
    """Keys follow first appearance and each list follows frame order, so a
    rerun on the same hydrofabric produces byte-identical logs."""
    frame = pd.DataFrame(
        {"toid": [9002, 9001, 9002, 9001]},
        index=pd.Index([400, 300, 200, 100], name="id"),
    )

    crosswalk = build_nexus_crosswalk(frame)

    assert list(crosswalk) == [9002, 9001]
    assert crosswalk == {9002: [400, 200], 9001: [300, 100]}


# --------------------------------------------------------------------- #
# The join is on toid, never on id                                      #
# --------------------------------------------------------------------- #
def test_nexus_keys_come_from_toid_not_from_the_index():
    """A flowpath whose *id* equals another row's nexus number: keying on the
    index instead of toid would produce a different, plausible answer."""
    frame = pd.DataFrame(
        # Feature 9001 is a flowpath id that collides with nexus 9001 below.
        {"toid": [500, 9001, 9001]},
        index=pd.Index([9001, 101, 102], name="id"),
    )

    crosswalk = build_nexus_crosswalk(frame)

    assert crosswalk == {500: [9001], 9001: [101, 102]}
    # The colliding flowpath is a *value* under nexus 500, and is not itself a
    # nexus key merely because some other flowpath drains to that number.
    assert crosswalk[500] == [9001]
    assert 9001 not in crosswalk[9001]


def test_a_flowpath_draining_to_itself_is_not_special_cased():
    """A self-referential toid is still read as a nexus number, not skipped."""
    frame = pd.DataFrame({"toid": [101]}, index=pd.Index([101], name="id"))

    assert build_nexus_crosswalk(frame) == {101: [101]}


def test_extra_columns_are_ignored(flowpaths_frame):
    """hydroseq, order and geometry ride along on the real frame and must not
    influence the mapping."""
    frame = flowpaths_frame.copy()
    frame["gage"] = None
    frame["order"] = 99

    assert build_nexus_crosswalk(frame) == {9001: [101, 102, 103], 9002: [201]}


# --------------------------------------------------------------------- #
# Absent or empty hydrofabric                                           #
# --------------------------------------------------------------------- #
def test_none_gives_an_empty_crosswalk():
    """A domain configured with no hydrofabric — load_hydrofabric's None path."""
    assert build_nexus_crosswalk(None) == {}


def test_empty_geodataframe_gives_an_empty_crosswalk():
    """Exactly what load_hydrofabric returns when gpkg_path is falsy: an empty
    GeoDataFrame with no columns at all, so the missing-toid check must not
    fire on it."""
    assert build_nexus_crosswalk(gpd.GeoDataFrame()) == {}


def test_empty_frame_with_a_toid_column_gives_an_empty_crosswalk():
    """A hydrofabric that loaded but held no flowpaths."""
    frame = pd.DataFrame({"toid": []}, index=pd.Index([], name="id"))

    assert build_nexus_crosswalk(frame) == {}


def test_empty_crosswalk_leaves_every_feature_uncovered(
    weight_frame, formulation_index_map, formulation_names, feature_ids
):
    """The empty result is usable, not merely empty: it resolves to equal
    weights under warn rather than raising."""
    resolved, report = resolve_weights(
        weight_frame,
        formulation_index_map,
        formulation_names,
        build_nexus_crosswalk(None),
        feature_ids,
        on_missing="warn",
    )

    assert report.covered_features == 0
    assert report.fraction == 0.0
    np.testing.assert_allclose(resolved.values, 1.0 / len(formulation_names))


# --------------------------------------------------------------------- #
# Prefixed identifiers, as a frame that skipped load_hydrofabric carries #
# --------------------------------------------------------------------- #
def test_prefixed_strings_reduce_to_the_same_crosswalk(flowpaths_frame):
    """``nex-9001`` and ``9001`` must not normalize differently, or the same
    hydrofabric would crosswalk two ways depending on how it was loaded."""
    prefixed = pd.DataFrame(
        {"toid": ["nex-9001", "nex-9001", "nex-9001", "nex-9002"]},
        index=pd.Index(["wb-101", "wb-102", "wb-103", "wb-201"], name="id"),
    )

    assert build_nexus_crosswalk(prefixed) == build_nexus_crosswalk(flowpaths_frame)


def test_terminal_nexus_prefix_is_reduced_by_digits():
    """``tnx-`` terminal nexuses differ from ``nex-`` only in the prefix, which
    is stripped, so they key on their number like any other nexus."""
    frame = pd.DataFrame(
        {"toid": ["tnx-77", "nex-88"]}, index=pd.Index([1, 2], name="id")
    )

    assert build_nexus_crosswalk(frame) == {77: [1], 88: [2]}


def test_mixed_prefixed_and_integer_toids_share_a_key():
    """One column holding both spellings still yields one group per nexus."""
    frame = pd.DataFrame(
        {"toid": ["nex-9001", 9001]}, index=pd.Index([101, 102], name="id")
    )

    assert build_nexus_crosswalk(frame) == {9001: [101, 102]}


def test_object_column_of_floats_is_not_read_by_stripping_the_decimal_point():
    """Digit-stripping 9001.0 would give 90010.  Numbers are read as numbers
    first, whatever dtype the column happens to hold."""
    frame = pd.DataFrame(
        {"toid": np.array([9001.0, 9001.0], dtype=object)},
        index=pd.Index([101, 102], name="id"),
    )
    assert frame["toid"].dtype == object

    assert build_nexus_crosswalk(frame) == {9001: [101, 102]}


def test_float_toid_column_reduces_to_integer_keys():
    """A float column (what a NaN forces pandas to) still keys on integers."""
    frame = pd.DataFrame(
        {"toid": [9001.0, 9002.0]}, index=pd.Index([101, 201], name="id")
    )

    crosswalk = build_nexus_crosswalk(frame)

    assert crosswalk == {9001: [101], 9002: [201]}
    assert all(type(key) is int for key in crosswalk)


# --------------------------------------------------------------------- #
# Malformed input                                                       #
# --------------------------------------------------------------------- #
def test_missing_toid_column_on_a_populated_frame_raises():
    """An empty frame with no columns is the no-hydrofabric case; a populated
    one without toid is a real error and must not pass as no coverage."""
    frame = pd.DataFrame({"hydroseq": [1, 2]}, index=pd.Index([101, 102], name="id"))

    with pytest.raises(ValueError, match="no 'toid' column"):
        build_nexus_crosswalk(frame)


def test_missing_toid_message_names_the_columns_found():
    """The message has to be diagnosable without opening the GeoPackage."""
    frame = pd.DataFrame({"hydroseq": [1]}, index=pd.Index([101], name="id"))

    with pytest.raises(ValueError, match="hydroseq"):
        build_nexus_crosswalk(frame)


def test_non_integral_toid_raises():
    """A fractional identifier is not a nexus; truncating it would invent one."""
    frame = pd.DataFrame({"toid": [9001.5]}, index=pd.Index([101], name="id"))

    with pytest.raises(ValueError, match="non-integer"):
        build_nexus_crosswalk(frame)


def test_non_integral_feature_id_raises():
    frame = pd.DataFrame({"toid": [9001]}, index=pd.Index([101.5], name="id"))

    with pytest.raises(ValueError, match="non-integer"):
        build_nexus_crosswalk(frame)


def test_boolean_toid_column_raises():
    """A bool column reads as 1/0 through to_numeric, which would file every
    flowpath under nexus 1."""
    frame = pd.DataFrame({"toid": [True, False]}, index=pd.Index([1, 2], name="id"))

    with pytest.raises(ValueError, match="boolean"):
        build_nexus_crosswalk(frame)


def test_boolean_toid_among_other_types_raises():
    """The same hazard hidden in an object column, which is not bool-dtyped:
    the check is on the values, so both sides of the join are guarded."""
    frame = pd.DataFrame(
        {"toid": [True, "nex-9001"]}, index=pd.Index([101, 102], name="id")
    )

    with pytest.raises(ValueError, match="boolean"):
        build_nexus_crosswalk(frame)


def test_missing_feature_id_raises():
    """A flowpath with no id cannot be addressed in the dataset at all."""
    frame = pd.DataFrame({"toid": [9001, 9002]}, index=pd.Index([101, None], name="id"))

    with pytest.raises(ValueError, match="no usable feature"):
        build_nexus_crosswalk(frame)


def test_feature_id_without_digits_raises():
    frame = pd.DataFrame({"toid": [9001]}, index=pd.Index(["wb-"], name="id"))

    with pytest.raises(ValueError, match="no usable feature"):
        build_nexus_crosswalk(frame)


# --------------------------------------------------------------------- #
# Flowpaths that cannot be placed                                       #
# --------------------------------------------------------------------- #
def test_null_toid_is_dropped_with_a_warning(caplog):
    """An unplaceable flowpath is left out rather than guessed at; it shows up
    downstream as an uncovered feature, which the coverage policy governs."""
    frame = pd.DataFrame(
        {"toid": [9001, None, 9001]}, index=pd.Index([101, 102, 103], name="id")
    )

    with caplog.at_level("WARNING"):
        crosswalk = build_nexus_crosswalk(frame)

    assert crosswalk == {9001: [101, 103]}
    assert "1 flowpath(s) carry no usable 'toid'" in caplog.text


def test_toid_without_digits_is_dropped():
    """A toid that reduces to nothing cannot be matched against a nexus."""
    frame = pd.DataFrame(
        {"toid": ["nex-", "nex-9001"]}, index=pd.Index([101, 102], name="id")
    )

    assert build_nexus_crosswalk(frame) == {9001: [102]}


def test_placeable_flowpaths_survive_an_unplaceable_neighbour(caplog):
    """Dropping one row must not shift the alignment of the others — the id
    comes from the index and the toid from a column, and they stay row-paired."""
    frame = pd.DataFrame(
        {"toid": [None, 9001, 9002, None, 9001]},
        index=pd.Index([100, 101, 201, 102, 103], name="id"),
    )

    with caplog.at_level("WARNING"):
        crosswalk = build_nexus_crosswalk(frame)

    assert crosswalk == {9001: [101, 103], 9002: [201]}
    assert "2 flowpath(s)" in caplog.text


def test_no_warning_when_every_flowpath_is_placed(flowpaths_frame, caplog):
    with caplog.at_level("WARNING"):
        build_nexus_crosswalk(flowpaths_frame)

    assert caplog.text == ""


# --------------------------------------------------------------------- #
# Duplicates and conflicts                                              #
# --------------------------------------------------------------------- #
def test_repeated_identical_rows_are_deduplicated():
    """A flowpath listed twice with the same toid is harmless; listing it twice
    in the group would make it look like two features at one nexus."""
    frame = pd.DataFrame(
        {"toid": [9001, 9001, 9001]}, index=pd.Index([101, 101, 102], name="id")
    )

    assert build_nexus_crosswalk(frame) == {9001: [101, 102]}


def test_a_feature_draining_to_two_nexuses_is_left_for_the_resolver():
    """The builder does not silently pick one. The ambiguity rule lives in the
    resolver, which rejects it there — asserted here so the two agree."""
    frame = pd.DataFrame(
        {"toid": [9001, 9002]}, index=pd.Index([101, 101], name="id")
    )

    crosswalk = build_nexus_crosswalk(frame)

    assert crosswalk == {9001: [101], 9002: [101]}


def test_the_resolver_rejects_the_ambiguous_crosswalk(
    weight_frame, formulation_index_map, formulation_names
):
    frame = pd.DataFrame(
        {"toid": [9001, 9002]}, index=pd.Index([101, 101], name="id")
    )

    with pytest.raises(ValueError, match="more than one nexus"):
        resolve_weights(
            weight_frame,
            formulation_index_map,
            formulation_names,
            build_nexus_crosswalk(frame),
            [101],
        )


# --------------------------------------------------------------------- #
# The builder reads nothing                                             #
# --------------------------------------------------------------------- #
def test_builds_without_reading_any_file(flowpaths_frame, monkeypatch):
    """The crosswalk comes off the already-loaded frame; a run must not pay a
    second GeoPackage read for it."""

    def fail(*args, **kwargs):
        raise AssertionError("build_nexus_crosswalk must not read any file")

    monkeypatch.setattr(gpd, "read_file", fail)
    monkeypatch.setattr(pd, "read_parquet", fail)

    assert build_nexus_crosswalk(flowpaths_frame) == {
        9001: [101, 102, 103],
        9002: [201],
    }


def test_the_input_frame_is_not_mutated(flowpaths_frame):
    """The caller keeps using this frame for metrics and mapping."""
    before = flowpaths_frame.copy(deep=True)

    build_nexus_crosswalk(flowpaths_frame)

    pd.testing.assert_frame_equal(flowpaths_frame, before)


# --------------------------------------------------------------------- #
# End to end with the resolver                                          #
# --------------------------------------------------------------------- #
def test_crosswalk_drives_the_resolver_to_hand_computed_weights(
    flowpaths_frame,
    weight_frame,
    formulation_index_map,
    formulation_names,
    feature_ids,
):
    """The two modules meet here: the confluence's three flowpaths receive the
    nexus 9001 group identically, and 201 receives the 9002 group."""
    resolved, report = resolve_weights(
        weight_frame,
        formulation_index_map,
        formulation_names,
        build_nexus_crosswalk(flowpaths_frame),
        feature_ids,
    )

    assert report.fraction == 1.0
    assert report.uncovered_features == 0
    for feature_id in (101, 102, 103):
        np.testing.assert_allclose(
            resolved.sel(feature_id=feature_id).values, [0.5, 0.3, 0.2]
        )
    np.testing.assert_allclose(
        resolved.sel(feature_id=201).values, [0.25, 0.75, 0.0]
    )


def test_partial_hydrofabric_leaves_the_rest_on_equal_weights(
    weight_frame, formulation_index_map, formulation_names, feature_ids
):
    """A hydrofabric that places only some of the run's features: the placed
    ones are weighted and the rest fall back, at the fraction the counts imply."""
    frame = pd.DataFrame(
        {"toid": [9001, 9001, 9001]}, index=pd.Index([101, 102, 103], name="id")
    )

    resolved, report = resolve_weights(
        weight_frame,
        formulation_index_map,
        formulation_names,
        build_nexus_crosswalk(frame),
        feature_ids,
    )

    assert report.covered_features == 3
    assert report.fraction == 0.75
    np.testing.assert_allclose(
        resolved.sel(feature_id=201).values, [1 / 3, 1 / 3, 1 / 3]
    )
