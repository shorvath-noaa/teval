"""
Shared pytest fixtures for the teval test suite.

The fixtures here provide the small synthetic inputs the ensemble machinery
operates on: a combined dataset carrying a ``formulation`` dimension, a
minimal flowpaths frame in the shape ``load_hydrofabric`` returns, and a tidy
weight frame in the provisional weight-file schema.

All values are chosen to be hand-checkable so tests can assert against
expectations worked out on paper rather than against a second implementation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr


# --------------------------------------------------------------------- #
# Formulations                                                          #
# --------------------------------------------------------------------- #
@pytest.fixture
def formulation_names():
    """Names of the synthetic ensemble members, in dataset order."""
    return ["formA", "formB", "formC"]


@pytest.fixture
def formulation_index_map():
    """1-based index to formulation name binding, as configuration carries it."""
    return {1: "formA", 2: "formB", 3: "formC"}


# --------------------------------------------------------------------- #
# Combined ensemble dataset                                             #
# --------------------------------------------------------------------- #
@pytest.fixture
def feature_ids():
    """Integer feature ids (prefix-stripped flowpath ids) of the synthetic run."""
    return [101, 102, 103, 201]


@pytest.fixture
def combined_ds(formulation_names, feature_ids):
    """
    Small lazy ensemble dataset shaped like an ``open_mfdataset`` result.

    Dimensions are ``(formulation, time, feature_id)`` with a single
    ``streamflow`` variable.  Values are ``formulation_offset + feature_offset
    + time_step`` so any reduction over ``formulation`` has an obvious
    closed-form expectation.

    The array is dask-backed, so tests can assert that graph construction
    stays lazy.
    """
    n_form = len(formulation_names)
    times = pd.date_range("2020-01-01", periods=4, freq="h")
    n_time = len(times)
    n_feat = len(feature_ids)

    form_offset = (np.arange(n_form) + 1)[:, None, None] * 100.0
    time_offset = np.arange(n_time)[None, :, None] * 1.0
    feat_offset = (np.arange(n_feat) + 1)[None, None, :] * 10.0
    values = form_offset + time_offset + feat_offset

    ds = xr.Dataset(
        {"streamflow": (("formulation", "time", "feature_id"), values)},
        coords={
            "formulation": formulation_names,
            "time": times,
            "feature_id": feature_ids,
        },
        attrs={"description": "synthetic ensemble"},
    )
    return ds.chunk({"formulation": 1, "time": -1, "feature_id": -1})


# --------------------------------------------------------------------- #
# Hydrofabric                                                           #
# --------------------------------------------------------------------- #
@pytest.fixture
def flowpaths_frame(feature_ids):
    """
    Minimal flowpaths frame in the shape ``load_hydrofabric`` returns.

    Indexed by integer ``id`` with an integer ``toid`` naming the downstream
    nexus.  Features 101, 102 and 103 converge on nexus 9001 (a confluence, so
    the nexus-to-feature relationship is genuinely many-to-one); 201 drains to
    its own nexus 9002.
    """
    return pd.DataFrame(
        {
            "toid": [9001, 9001, 9001, 9002],
            "hydroseq": [4, 3, 2, 1],
            "order": [1, 1, 2, 1],
        },
        index=pd.Index(feature_ids, name="id"),
    )


# --------------------------------------------------------------------- #
# Weights                                                               #
# --------------------------------------------------------------------- #
@pytest.fixture
def weight_frame():
    """
    Tidy weight frame in the provisional weight-file schema.

    Columns are ``nexus_id`` (string, ``nex-`` prefix retained),
    ``formulation_index`` (1-based int) and ``weight`` (float).  Both groups
    are complete over the three formulations and sum to 1; nexus 9002 carries
    an individual zero weight, which is permitted.
    """
    return pd.DataFrame(
        {
            "nexus_id": ["nex-9001"] * 3 + ["nex-9002"] * 3,
            "formulation_index": [1, 2, 3, 1, 2, 3],
            "weight": [0.5, 0.3, 0.2, 0.25, 0.75, 0.0],
        }
    )
