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
    Written only when weighting was applied.
``ensemble_weight_coverage_fraction``
    The fraction of the domain's features that carried weights the file
    supplied; the remainder fell back to equal weights.  Written only when
    weighting was applied.  A value below 1.0 is the signal that the mean is
    weighted in part of the domain and plain in the rest.

Values are strings and floats rather than booleans because NetCDF has no
boolean attribute type; ``"true"``/``"false"`` survives the round trip
through the file where a Python ``bool`` would not.

Public API
----------
weighting_attrs(config=None, report=None)
    Build the attribute mapping for a run, weighted or not.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from teval.config import WeightsConfig
from teval.weights.resolve import CoverageReport

#: Whether the mean in this file is a weighted combination of the members.
APPLIED_ATTR = "ensemble_weighting_applied"
#: The weight file that produced it, when one did.
FILE_ATTR = "ensemble_weight_file"
#: How much of the domain that file actually reached.
COVERAGE_ATTR = "ensemble_weight_coverage_fraction"

#: The two values ``APPLIED_ATTR`` takes.  NetCDF has no boolean attribute.
APPLIED_TRUE = "true"
APPLIED_FALSE = "false"


def weighting_attrs(
    config: Optional[WeightsConfig] = None,
    report: Optional[CoverageReport] = None,
) -> Dict[str, Any]:
    """
    Build the provenance attributes describing how the mean was combined.

    Parameters
    ----------
    config:
        The ``stats.weights`` block that was applied, or ``None`` for an
        unweighted run.
    report:
        The coverage the resolution achieved, as ``resolve_weights`` returned
        it, or ``None`` for an unweighted run.

    Returns
    -------
    dict
        ``{APPLIED_ATTR: "false"}`` for an unweighted run; that key set to
        ``"true"`` alongside ``FILE_ATTR`` and ``COVERAGE_ATTR`` for a
        weighted one.  The file path and the coverage fraction are omitted
        rather than zero-filled when nothing was applied, since a recorded
        coverage of 0.0 would read as "weighting was attempted and reached
        nothing", which is a different — and much worse — outcome.

    Raises
    ------
    ValueError
        Exactly one of *config* and *report* is given.  The pair travels
        together out of the resolution step, so a lone one means the caller
        lost track of which run it is describing, and guessing would write
        provenance that misdescribes the file.
    """
    if (config is None) != (report is None):
        raise ValueError(
            "weighting_attrs needs both the weights configuration and the "
            "coverage report, or neither: a weighted run has both and an "
            f"unweighted run has neither; got config={config!r}, "
            f"report={report!r}."
        )

    if config is None:
        return {APPLIED_ATTR: APPLIED_FALSE}

    return {
        APPLIED_ATTR: APPLIED_TRUE,
        FILE_ATTR: str(config.file),
        COVERAGE_ATTR: float(report.fraction),
    }
