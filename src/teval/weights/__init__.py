"""
teval.weights

Spatially varying ensemble weights.

Weights are supplied per nexus and per formulation: one group of weights per
nexus, carrying one weight for each ensemble member.  Reading a weight file
is kept separate from interpreting it, so the provisional file format can be
replaced without touching the validation and expansion rules.

Submodules
----------
reader      — read a weight file into a tidy, dtype-validated DataFrame
resolve     — relabel the file's indices to formulation names, enforce the
              weight rules, and expand per-nexus groups into a dense array
              over the run's features
plan        — drive the two phases of a weighted domain: gather the file and
              the crosswalk from the hydrofabric, then resolve them against
              the opened ensemble
provenance  — the attributes recording, in the output file itself, whether
              weighting was applied and how far it reached
"""

from teval.weights.plan import (
    WeightPlan,
    prepare_weight_plan,
    resolve_domain_weights,
)
from teval.weights.provenance import AppliedWeighting, weighting_attrs
from teval.weights.reader import read_weight_file
from teval.weights.resolve import (
    CoverageReport,
    resolve_weights,
    validate_weight_groups,
)

__all__ = [
    "read_weight_file",
    "validate_weight_groups",
    "resolve_weights",
    "CoverageReport",
    "WeightPlan",
    "prepare_weight_plan",
    "resolve_domain_weights",
    "AppliedWeighting",
    "weighting_attrs",
]
