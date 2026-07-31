"""
teval.weights.provenance

Attributes recording whether ensemble weighting was applied, and with what.

A weighted mean and an unweighted mean are the same shape, the same dtype and
the same variable name, and for a domain with poor weight coverage they are
numerically close as well.  Nothing in the values themselves says which one a
file holds.  These attributes make the two distinguishable from the output
alone, without the run's configuration or its log:

``ensemble_weighting_applied``
    ``"true"`` or ``"false"`` — always written, so a file that omits it was
    produced before weighting existed rather than by an unweighted run.
``ensemble_weight_file``
    The configured weight file, verbatim as the configuration named it.
``ensemble_weight_coverage_fraction``
    The fraction of the domain's features that carried weights the file
    supplied; the remainder fell back to equal weights.  Below 1.0 is the
    signal that the mean is weighted in part of the domain and plain in the
    rest.

The latter two are written only when weighting was applied.  All three are
strings and floats rather than booleans because NetCDF has no boolean
attribute type.

Public API
----------
AppliedWeighting
    What a weighted run applied: the configuration and the coverage reached.
weighting_attrs(applied=None)
    Build the attribute mapping for a run, weighted or not.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from teval.config import WeightsConfig
from teval.weights.resolve import CoverageReport

#: The attribute names, each described in the module docstring above.
APPLIED_ATTR = "ensemble_weighting_applied"
FILE_ATTR = "ensemble_weight_file"
COVERAGE_ATTR = "ensemble_weight_coverage_fraction"

#: The two values ``APPLIED_ATTR`` takes.
APPLIED_TRUE = "true"
APPLIED_FALSE = "false"


@dataclass(frozen=True)
class AppliedWeighting:
    """
    The ``stats.weights`` block a run applied and the coverage it reached.

    Resolution produces the two together and they describe one event between
    them, so they travel as one value: a run either has this or has ``None``,
    and cannot hold half of it.
    """

    config: WeightsConfig
    report: CoverageReport


def weighting_attrs(applied: Optional[AppliedWeighting] = None) -> Dict[str, Any]:
    """
    Build the provenance attributes for a run, weighted (*applied*) or not.

    An unweighted run records only that it was unweighted: the file and the
    coverage are omitted rather than zero-filled, since a recorded coverage of
    0.0 would read as "weighting was attempted and reached nothing", a
    different and much worse outcome.
    """
    if applied is None:
        return {APPLIED_ATTR: APPLIED_FALSE}

    return {
        APPLIED_ATTR: APPLIED_TRUE,
        FILE_ATTR: str(applied.config.file),
        COVERAGE_ATTR: float(applied.report.fraction),
    }
