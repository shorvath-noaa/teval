"""
Smoke tests proving the suite collects and runs, and that the shared fixtures
are wired up as the rest of the suite expects.
"""

from __future__ import annotations

import pandas as pd


def test_suite_runs():
    """The suite collects and executes."""
    assert True


def test_teval_imports():
    """The package under test is importable from the test environment."""
    import teval

    assert teval is not None


def test_combined_ds_fixture_shape(combined_ds, formulation_names, feature_ids):
    """The synthetic ensemble carries a formulation dimension and stays lazy."""
    assert combined_ds.sizes["formulation"] == len(formulation_names)
    assert list(combined_ds["feature_id"].values) == feature_ids
    assert combined_ds["streamflow"].chunks is not None


def test_flowpaths_frame_fixture_has_confluence(flowpaths_frame):
    """The flowpaths fixture exercises the many-to-one nexus relationship."""
    assert flowpaths_frame.index.name == "id"
    assert "toid" in flowpaths_frame.columns
    assert (flowpaths_frame["toid"].value_counts() > 1).any()


def test_weight_frame_fixture_schema(weight_frame, formulation_index_map):
    """The weight fixture is tidy, prefix-retaining, and sums to one per group."""
    assert list(weight_frame.columns) == ["nexus_id", "formulation_index", "weight"]
    assert weight_frame["nexus_id"].str.startswith("nex-").all()

    group_sums = weight_frame.groupby("nexus_id")["weight"].sum()
    assert group_sums.round(9).eq(1.0).all()

    per_group_indices = weight_frame.groupby("nexus_id")["formulation_index"].apply(
        lambda s: sorted(s)
    )
    expected = sorted(formulation_index_map)
    assert all(indices == expected for indices in per_group_indices)

    assert pd.api.types.is_numeric_dtype(weight_frame["weight"])
