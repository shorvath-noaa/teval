"""
teval.weights

Spatially varying ensemble weights.

Weights are supplied per nexus and per formulation: one group of weights per
nexus, carrying one weight for each ensemble member.  Reading a weight file
is kept separate from interpreting it, so the provisional file format can be
replaced without touching the validation and expansion rules.

Submodules
----------
reader  — read a weight file into a tidy, dtype-validated DataFrame
"""

from teval.weights.reader import read_weight_file

__all__ = [
    "read_weight_file",
]
