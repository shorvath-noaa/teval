"""
teval.weights

Spatially varying ensemble weights.

Weights are supplied per nexus and per formulation: one group of weights per
nexus, carrying one weight for each ensemble member.  Reading a weight file
is kept separate from interpreting it, so the provisional file format can be
replaced without touching the validation and expansion rules.

Submodules
----------
reader   — read a weight file into a tidy, dtype-validated DataFrame
resolve  — bind indices to formulation names, enforce the weight rules, and
           expand per-nexus groups into a dense array over the run's features
"""

from teval.weights.reader import read_weight_file
from teval.weights.resolve import (
    CoverageReport,
    bind_formulation_indices,
    resolve_weights,
    validate_weight_groups,
)

__all__ = [
    "read_weight_file",
    "bind_formulation_indices",
    "validate_weight_groups",
    "resolve_weights",
    "CoverageReport",
]
