"""
teval.weights.resolve

Interpret a tidy weight frame: bind the file's integer formulation indices to
the run's formulation names, and enforce every rule a weight group must obey.

This module is a pure function of plain inputs — DataFrames, dicts and
sequences.  It reads no file, opens no GeoPackage and touches no xarray
Dataset, so every rule below is testable without any of them.  Reading the
file is ``teval.weights.reader``'s job; that module owns the provisional file
format and this one owns the meaning, so a format change leaves these rules
intact.

Rules enforced here
-------------------
Binding
    ``formulation_index_map`` must be a bijection with the formulations
    discovered in the run.  An index naming a formulation that is not in the
    run, or a formulation in the run that no index names, is an error: the
    first is a stale legend, the second would silently drop a member from the
    product.
Group completeness
    A nexus with any rows at all must carry exactly one row per configured
    index.  A missing row would drop one member at one location; a duplicate
    row is the signature of two weight files concatenated by accident.  Both
    are hard errors and neither is configurable.  A nexus with *no* rows is
    not an error here — that is coverage, governed by ``on_missing``.
Sign
    Negative weights are an error, so a sign error upstream cannot produce a
    physically meaningless combination.
All-zero groups
    A group whose weights are all zero is an error — that location would
    silently produce zero flow.  An individual zero inside an otherwise
    non-zero group is permitted and meaningful: it excludes one formulation at
    one location deliberately.
Sums
    Each group must sum to 1 within ``SUM_TOLERANCE`` so float representation
    error does not reject a legitimate file.  With ``normalize`` set, each
    group is divided by its sum instead, accepting any positive scale.

Every failure raises ``ValueError`` naming the offending nexus ids, so a bad
file is diagnosable from the message without reading the source.

Public API
----------
bind_formulation_indices(formulation_index_map, formulations)
    Return the weight-file index for each formulation, in run order.
validate_weight_groups(weights, formulation_index_map, formulations, ...)
    Return validated per-nexus weight groups as a wide frame whose rows sum
    to 1 and whose columns are the formulations in run order.
"""

from __future__ import annotations

import logging
from typing import List, Mapping, Sequence

import numpy as np
import pandas as pd

from teval.weights.reader import REQUIRED_COLUMNS

logger = logging.getLogger(__name__)

#: How far a group's sum may sit from 1.0 before it is rejected.  Wide enough
#: that 0.5 + 0.3 + 0.2 passes, narrow enough that a real error does not.
SUM_TOLERANCE = 1e-6

# How many offending nexus ids to name in an error message before truncating.
_MAX_REPORTED = 10


def _describe(items) -> str:
    """Render a collection of identifiers as a short, truncated list."""
    listed = list(items)[:_MAX_REPORTED]
    text = ", ".join(str(item) for item in listed)
    remaining = len(items) - len(listed)
    return f"{text} (and {remaining} more)" if remaining else text or "(none)"


def bind_formulation_indices(
    formulation_index_map: Mapping[int, str],
    formulations: Sequence[str],
) -> List[int]:
    """
    Bind the weight file's integer indices to the run's formulation names.

    The ``formulation`` dimension of the combined dataset follows directory
    scan order, which is arbitrary and not stable across machines, so an index
    cannot be read positionally.  The binding is therefore explicit, and it
    must be a bijection: exactly the formulations discovered in the run, each
    named exactly once.

    Parameters
    ----------
    formulation_index_map:
        Mapping from the 1-based ``formulation_index`` used in the weight file
        to the formulation name teval parsed from the run directories.
    formulations:
        The formulation names discovered in the run, in dataset order.

    Returns
    -------
    list of int
        The weight-file index of each formulation, positionally aligned with
        ``formulations``, so the result can order a weight array to match the
        dataset's ``formulation`` dimension.

    Raises
    ------
    ValueError
        The map or the formulation list is empty or carries a duplicate, the
        map names a formulation absent from the run, or a formulation in the
        run is absent from the map.
    """
    if not formulation_index_map:
        raise ValueError(
            "formulation_index_map is empty; it must bind every formulation "
            "in the run to the index the weight file uses for it."
        )
    if not formulations:
        raise ValueError(
            "No formulations were supplied to bind weights against; the run "
            "must carry at least one formulation."
        )

    run_names = list(formulations)
    repeated_in_run = sorted({n for n in run_names if run_names.count(n) > 1})
    if repeated_in_run:
        raise ValueError(
            f"Formulation names must be unique; the run repeats: "
            f"{_describe(repeated_in_run)}."
        )

    mapped_names = list(formulation_index_map.values())
    repeated_in_map = sorted({n for n in mapped_names if mapped_names.count(n) > 1})
    if repeated_in_map:
        raise ValueError(
            f"formulation_index_map must name each formulation at most once; "
            f"duplicated: {_describe(repeated_in_map)}."
        )

    # Both directions of the bijection are checked, and reported together, so
    # a legend that is wrong in both directions takes one run to diagnose.
    unknown = sorted(set(mapped_names) - set(run_names))
    unmapped = sorted(set(run_names) - set(mapped_names))
    if unknown or unmapped:
        problems = []
        if unknown:
            problems.append(
                f"formulation_index_map names formulation(s) not present in "
                f"the run: {_describe(unknown)}"
            )
        if unmapped:
            problems.append(
                f"formulation(s) present in the run are missing from "
                f"formulation_index_map: {_describe(unmapped)}"
            )
        raise ValueError(
            f"{'; '.join(problems)}. The run's formulations are "
            f"{_describe(run_names)} and the map names {_describe(sorted(mapped_names))}. "
            f"Weighting requires the two to match exactly."
        )

    index_by_name = {name: int(index) for index, name in formulation_index_map.items()}
    return [index_by_name[name] for name in run_names]


def _as_tidy_frame(weights: pd.DataFrame) -> pd.DataFrame:
    """
    Return the three schema columns with the dtypes the rules assume.

    ``read_weight_file`` already guarantees these, but the resolver is a pure
    function that callers and tests may hand a frame built by other means, so
    the assumption is enforced rather than trusted.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in weights.columns]
    if missing:
        raise ValueError(
            f"Weight frame is missing required column(s): "
            f"{', '.join(missing)}. Found: "
            f"{', '.join(str(c) for c in weights.columns) or '(none)'}."
        )

    tidy = weights[list(REQUIRED_COLUMNS)].reset_index(drop=True)
    try:
        return pd.DataFrame(
            {
                "nexus_id": tidy["nexus_id"].astype(str).str.strip(),
                "formulation_index": tidy["formulation_index"].astype(int),
                "weight": tidy["weight"].astype(float),
            }
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Weight frame columns could not be read as "
            f"nexus_id (str), formulation_index (int), weight (float): {exc}"
        ) from exc


def _require_known_indices(tidy: pd.DataFrame, indices: Sequence[int]) -> None:
    """Raise if the file uses an index the legend does not define."""
    unknown = sorted(set(tidy["formulation_index"]) - set(indices))
    if unknown:
        offenders = tidy.loc[tidy["formulation_index"].isin(unknown), "nexus_id"]
        raise ValueError(
            f"Weight file uses formulation_index value(s) "
            f"{_describe(unknown)} that formulation_index_map does not "
            f"define (it defines {_describe(sorted(indices))}), at nexus "
            f"{_describe(sorted(set(offenders)))}."
        )


def _require_complete_groups(tidy: pd.DataFrame, indices: Sequence[int]) -> None:
    """
    Raise unless every nexus present carries exactly one row per index.

    A missing row would drop one formulation from the combination at that
    location; a duplicate row makes the group ambiguous.  Both are reported
    with the nexus ids and the indices at fault.
    """
    expected = set(indices)

    duplicated = tidy[tidy.duplicated(subset=["nexus_id", "formulation_index"])]
    if not duplicated.empty:
        pairs = sorted(
            {
                (row.nexus_id, int(row.formulation_index))
                for row in duplicated.itertuples()
            }
        )
        raise ValueError(
            f"Weight file has duplicate rows for "
            f"{_describe([f'{nexus} index {index}' for nexus, index in pairs])}. "
            f"Each nexus must carry exactly one row per formulation index; "
            f"duplicates usually mean two weight files were concatenated."
        )

    incomplete = []
    for nexus_id, group in tidy.groupby("nexus_id", sort=False):
        missing = sorted(expected - set(group["formulation_index"]))
        if missing:
            incomplete.append(f"{nexus_id} missing index/indices {missing}")
    if incomplete:
        raise ValueError(
            f"Weight file has incomplete weight group(s): "
            f"{_describe(incomplete)}. A nexus present in the file must carry "
            f"a weight for every formulation index in formulation_index_map "
            f"({sorted(expected)}), so no member can silently drop out at one "
            f"location. A nexus with no rows at all is allowed and is handled "
            f"by the coverage policy."
        )


def _require_valid_weight_values(tidy: pd.DataFrame) -> None:
    """Raise on non-finite or negative weights, naming the nexus ids."""
    weight = tidy["weight"]

    non_finite = tidy.loc[~np.isfinite(weight), "nexus_id"]
    if not non_finite.empty:
        raise ValueError(
            f"Weight file has non-finite weights at nexus "
            f"{_describe(sorted(set(non_finite)))}. Every weight must be a "
            f"finite, non-negative number."
        )

    negative = tidy.loc[weight < 0, "nexus_id"]
    if not negative.empty:
        raise ValueError(
            f"Weight file has negative weights at nexus "
            f"{_describe(sorted(set(negative)))} (minimum {weight.min()}). "
            f"Negative weights are not physically meaningful in an ensemble "
            f"combination."
        )


def _to_wide(
    tidy: pd.DataFrame,
    indices: Sequence[int],
    formulations: Sequence[str],
) -> pd.DataFrame:
    """
    Pivot validated rows into one row per nexus, columns in run order.

    Completeness and duplication have already been checked, so the pivot is
    guaranteed dense and unambiguous.
    """
    order = tidy["nexus_id"].drop_duplicates().tolist()
    wide = tidy.pivot(index="nexus_id", columns="formulation_index", values="weight")
    wide = wide.loc[order, list(indices)]
    wide.columns = pd.Index(list(formulations), name="formulation")
    return wide.astype(float)


def _apply_sum_rule(
    wide: pd.DataFrame,
    normalize: bool,
    tolerance: float,
) -> pd.DataFrame:
    """
    Reject all-zero groups, then either normalize or enforce the sum-to-one rule.

    The all-zero check runs first so a group of zeros is reported as what it
    is rather than as a group summing to 0 instead of 1.
    """
    sums = wide.sum(axis=1)

    all_zero = sums.index[sums == 0].tolist()
    if all_zero:
        raise ValueError(
            f"Weight file has all-zero weight group(s) at nexus "
            f"{_describe(all_zero)}. A group of zeros would produce zero flow "
            f"at that location. Individual zero weights are permitted, but at "
            f"least one weight in a group must be non-zero."
        )

    if normalize:
        logger.debug(
            f"Normalizing {len(wide)} weight group(s); observed sums span "
            f"{sums.min():.6g} to {sums.max():.6g}."
        )
        return wide.div(sums, axis=0)

    off = (sums - 1.0).abs() > tolerance
    if off.any():
        offenders = [
            f"{nexus} sums to {sums[nexus]:.9g}" for nexus in sums.index[off].tolist()
        ]
        raise ValueError(
            f"Weight group(s) do not sum to 1 within {tolerance:g}: "
            f"{_describe(offenders)}. Fix the file, or set "
            f"stats.weights.normalize to divide each group by its own sum."
        )
    return wide


def validate_weight_groups(
    weights: pd.DataFrame,
    formulation_index_map: Mapping[int, str],
    formulations: Sequence[str],
    normalize: bool = False,
    tolerance: float = SUM_TOLERANCE,
) -> pd.DataFrame:
    """
    Bind, validate and normalize the weight groups in a tidy weight frame.

    Pure: plain frames and dicts in, a plain frame out.  No file is read and
    no dataset is touched, so this runs before any Dask graph is built and a
    bad weight file fails before the expensive compute.

    Parameters
    ----------
    weights:
        Tidy weight frame with ``nexus_id``, ``formulation_index`` and
        ``weight`` columns, as ``read_weight_file`` returns it.
    formulation_index_map:
        Binding from weight-file index to formulation name.
    formulations:
        The formulation names discovered in the run, in dataset order.  The
        returned columns follow this order.
    normalize:
        If True, divide each group by its own sum, accepting any positive
        scale.  If False (default), each group must already sum to 1 within
        ``tolerance``.
    tolerance:
        How far a group's sum may sit from 1.0 when ``normalize`` is False.

    Returns
    -------
    pd.DataFrame
        One row per nexus present in the file, indexed by ``nexus_id`` in
        order of first appearance, with one float column per formulation
        named and ordered as ``formulations``.  Every row sums to 1 within
        ``tolerance``.  A frame with no rows in returns an empty frame with
        those columns — an absent nexus is a coverage question, not a
        validation failure.

    Raises
    ------
    ValueError
        The index map is not a bijection with the run's formulations, the
        frame is missing a schema column or carries values of the wrong type,
        a group is incomplete or duplicated, a weight is negative or
        non-finite, a group is entirely zero, or a group does not sum to 1
        and ``normalize`` is False.
    """
    indices = bind_formulation_indices(formulation_index_map, formulations)
    tidy = _as_tidy_frame(weights)

    if tidy.empty:
        logger.warning(
            "Weight frame carries no rows; no nexus is covered by weights."
        )
        return pd.DataFrame(
            columns=pd.Index(list(formulations), name="formulation"),
            index=pd.Index([], name="nexus_id", dtype=object),
            dtype=float,
        )

    _require_known_indices(tidy, indices)
    _require_complete_groups(tidy, indices)
    _require_valid_weight_values(tidy)

    wide = _to_wide(tidy, indices, formulations)
    wide = _apply_sum_rule(wide, normalize=normalize, tolerance=tolerance)

    logger.debug(
        f"Validated {len(wide)} weight group(s) over "
        f"{len(formulations)} formulation(s)."
    )
    return wide
