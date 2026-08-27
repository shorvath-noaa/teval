"""
Helpers shared by the weighting tests.

Not a test module — pytest collects ``test_*.py`` only, so nothing here runs on
its own.  It holds the plain functions and constants that more than one
weighting test module needs; anything used by a single module stays in that
module, and anything that needs pytest's fixture machinery lives in
``conftest.py``.

Two levels are represented, kept apart by the banners below: builders for the
tidy weight frames the resolver consumes, and the harness the ``load_domain_data``
tests drive the whole pipeline with.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from teval.config import StatsConfig, WeightsConfig

RESOLVE_LOGGER = "teval.weights.resolve"
PLAN_LOGGER = "teval.weights.plan"
WORKFLOW_LOGGER = "teval.workflow"


# --------------------------------------------------------------------- #
# Tidy weight frames                                                    #
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


# --------------------------------------------------------------------- #
# The load_domain_data harness                                          #
# --------------------------------------------------------------------- #
#: The per-feature member part of the weighted mean, worked out on paper.
WEIGHTED_MEMBER_PART = {101: 170.0, 102: 170.0, 103: 170.0, 201: 175.0}
#: What a feature falling back to equal weights gets, which is the plain mean.
EQUAL_MEMBER_PART = 200.0
#: The same, for every feature -- what an unweighted run must produce.
ALL_EQUAL_MEMBER_PARTS = dict.fromkeys(WEIGHTED_MEMBER_PART, EQUAL_MEMBER_PART)


def domain_map(raw_files, ensemble_file=None):
    """A domain map entry in the shape ``initialize_domains`` produces."""
    return {
        "formulations": {"raw_files": raw_files, "ensemble_file": ensemble_file},
        "hydrofabric": Path("domain.gpkg"),
        "gage_obs": {"domain_name": [], "obs_file": [None]},
    }


def weighted_config(weight_file, formulation_index_map, **overrides):
    """A ``StatsConfig`` carrying a weights block."""
    return StatsConfig(
        weights=WeightsConfig(
            file=weight_file,
            formulation_index_map=formulation_index_map,
            **overrides,
        )
    )


def expected_mean(member_part_by_feature, ds_stats):
    """
    Build the expected ``(time, feature_id)`` mean from the fixture's closed form.

    The whole array is reconstructed rather than a spot value, so a
    misalignment on either axis has nowhere to hide.
    """
    times = np.arange(ds_stats.sizes["time"], dtype=float)
    features = [int(f) for f in ds_stats.feature_id.values]
    feature_offset = {101: 10.0, 102: 20.0, 103: 30.0, 201: 40.0}
    return np.array(
        [
            [member_part_by_feature[f] + t + feature_offset[f] for f in features]
            for t in times
        ]
    )


def mean_of(results):
    """The stats mean as a ``(time, feature_id)`` array."""
    ds_stats = results["formulations"]["combined"]
    return ds_stats, ds_stats.streamflow_mean.compute().transpose(
        "time", "feature_id"
    ).values
