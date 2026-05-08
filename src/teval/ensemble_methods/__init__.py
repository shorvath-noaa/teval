"""
teval.ensemble_methods

Core ensemble combination utilities for the teval pipeline.

This package contains the base abstractions and statistical helpers used
during ensemble stat computation.  Performance-weighted and other advanced
combination methods are in ``teval.experimental``.
"""

from teval.ensemble_methods.base import (
    SEASONS,
    KGE_SKILL_THRESHOLD,
    EnsembleMethod,
    get_season,
    assign_seasons,
    compute_seasonal_kge,
    apply_skill_threshold,
    softmax_weights,
)
from teval.ensemble_methods.stats import build_stats

__all__ = [
    "SEASONS",
    "KGE_SKILL_THRESHOLD",
    "EnsembleMethod",
    "get_season",
    "assign_seasons",
    "compute_seasonal_kge",
    "apply_skill_threshold",
    "softmax_weights",
    "build_stats",
]
