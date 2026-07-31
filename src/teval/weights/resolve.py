"""
teval.weights.resolve

Interpret a tidy weight frame: relabel the file's integer formulation indices
to the run's formulation names, enforce every rule a weight group must obey,
and expand the per-nexus groups into a dense weight array over the run's
features.

This module is a pure function of plain inputs — DataFrames, dicts and
sequences.  It reads no file, opens no GeoPackage and touches no xarray
Dataset, so every rule below is testable without any of them.  Reading the
file is ``teval.weights.reader``'s job; that module owns the provisional file
format and this one owns the meaning, so a format change leaves these rules
intact.

Because nothing here touches data, everything — validation *and* coverage —
is decided before a Dask graph exists.  A weight file that cannot be applied
therefore fails in the first second of a run rather than after a long compute.

The file's ``formulation_index`` is a file-format detail and does not survive
past the first step here: rows are relabelled to formulation names once, at
the top of :func:`validate_weight_groups`, and every rule below is expressed
and reported in name space, which is what a user reading an error message can
act on.

Rules enforced here
-------------------
Legend
    ``formulation_index_map`` must name exactly the formulations discovered in
    the run.  A name the map supplies that the run does not have is a stale
    legend; a formulation in the run that the map does not name would silently
    drop a member from the product.  One set comparison decides both.
Group completeness
    A nexus with any rows at all must carry exactly one row per formulation in
    the run.  A missing row would drop one member at one location; a duplicate
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

Expansion and coverage
----------------------
Weights are keyed by nexus; the ensemble dataset is indexed by ``feature_id``.
The relationship is many-to-one — several flowpaths may converge on one nexus
— so a nexus' group broadcasts unchanged to every feature draining to it.

A feature whose nexus carries no weights is *uncovered*, and ``on_missing``
decides what that means: ``warn`` gives it equal weights (which is exactly the
simple mean, so an uncovered feature behaves as it did before weighting was
configured) and logs the counts and the coverage fraction; ``error`` aborts.

Public API
----------
validate_weight_groups(weights, formulation_index_map, formulations, ...)
    Return validated per-nexus weight groups as a wide frame whose rows sum
    to 1 and whose columns are the formulations in run order.
resolve_weights(weights, formulation_index_map, formulations, ...)
    Validate, expand and fill in one call: return a dense weight array over
    (feature_id, formulation) together with a CoverageReport.
CoverageReport
    Counts and the coverage fraction achieved by a resolution.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import xarray as xr

from teval.identifiers import as_identifiers
from teval.weights.reader import REQUIRED_COLUMNS

logger = logging.getLogger(__name__)

#: How far a group's sum may sit from 1.0 before it is rejected.  Wide enough
#: that 0.5 + 0.3 + 0.2 passes, narrow enough that a real error does not.
SUM_TOLERANCE = 1e-6

#: Accepted values of ``stats.weights.on_missing``.
ON_MISSING_POLICIES = ("warn", "error")

# How many offending nexus ids to name in an error message before truncating.
_MAX_REPORTED = 10


def _describe(items) -> str:
    """Render a collection of identifiers as a short, truncated list."""
    listed = list(items)[:_MAX_REPORTED]
    text = ", ".join(str(item) for item in listed)
    remaining = len(items) - len(listed)
    return f"{text} (and {remaining} more)" if remaining else text or "(none)"


def _require_legend_matches_run(
    formulation_index_map: Mapping[int, str],
    formulations: Sequence[str],
) -> None:
    """
    Require the legend to name exactly the formulations the run discovered.

    One set comparison decides it, in both directions and reported together, so
    a legend wrong in both takes one run to diagnose.  A name the map supplies
    that the run does not have is a stale legend; a formulation in the run that
    the map does not name would silently drop a member from the product.  An
    empty map, or one that spends two indices on the same name and so leaves
    another formulation unnamed, fails here as an unmapped formulation.

    The ``formulation`` dimension of the combined dataset follows directory
    scan order, which is arbitrary and not stable across machines, which is why
    the legend is explicit rather than positional in the first place.
    """
    run_names = list(formulations)
    if not run_names:
        raise ValueError(
            "No formulations were supplied to weight against; the run must "
            "carry at least one formulation."
        )
    repeated_in_run = sorted({n for n in run_names if run_names.count(n) > 1})
    if repeated_in_run:
        raise ValueError(
            f"Formulation names must be unique; the run repeats: "
            f"{_describe(repeated_in_run)}."
        )

    mapped_names = set(formulation_index_map.values())
    if mapped_names == set(run_names):
        return

    problems = []
    unknown = sorted(mapped_names - set(run_names))
    if unknown:
        problems.append(
            f"formulation_index_map names formulation(s) not present in "
            f"the run: {_describe(unknown)}"
        )
    unmapped = sorted(set(run_names) - mapped_names)
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


def _relabel(
    tidy: pd.DataFrame,
    formulation_index_map: Mapping[int, str],
) -> pd.DataFrame:
    """
    Replace the file's integer indices with formulation names, once.

    This is the only place the ``formulation_index`` is understood.  Past it
    every rule, every pivot and every error message works in name space, so
    nothing downstream has to explain the legend to whoever reads the message.

    An index the legend does not define is an error rather than a dropped row:
    the file supplied a weight for a member and teval would otherwise ignore
    it, leaving the group both silently incomplete and silently renormalized.
    """
    by_index = {int(index): name for index, name in formulation_index_map.items()}
    named = tidy["formulation_index"].map(by_index)

    unknown = tidy.loc[named.isna()]
    if not unknown.empty:
        raise ValueError(
            f"Weight file uses formulation_index value(s) "
            f"{_describe(sorted(set(unknown['formulation_index'])))} that "
            f"formulation_index_map does not define (it defines "
            f"{_describe(sorted(by_index))}), at nexus "
            f"{_describe(sorted(set(unknown['nexus_id'])))}."
        )
    return tidy.assign(formulation=named).drop(columns="formulation_index")


def _require_complete_groups(tidy: pd.DataFrame, formulations: Sequence[str]) -> None:
    """
    Raise unless every nexus present carries exactly one row per formulation.

    A missing row would drop one formulation from the combination at that
    location; a duplicate row makes the group ambiguous.  Both are reported
    with the nexus ids and the formulations at fault.
    """
    expected = set(formulations)

    duplicated = tidy[tidy.duplicated(subset=["nexus_id", "formulation"])]
    if not duplicated.empty:
        pairs = sorted(
            {(row.nexus_id, row.formulation) for row in duplicated.itertuples()}
        )
        raise ValueError(
            f"Weight file has duplicate rows for "
            f"{_describe([f'{nexus} formulation {name}' for nexus, name in pairs])}. "
            f"Each nexus must carry exactly one row per formulation; "
            f"duplicates usually mean two weight files were concatenated."
        )

    incomplete = []
    for nexus_id, group in tidy.groupby("nexus_id", sort=False):
        missing = sorted(expected - set(group["formulation"]))
        if missing:
            incomplete.append(f"{nexus_id} missing {', '.join(missing)}")
    if incomplete:
        raise ValueError(
            f"Weight file has incomplete weight group(s): "
            f"{_describe(incomplete)}. A nexus present in the file must carry "
            f"a weight for every formulation in the run "
            f"({_describe(sorted(expected))}), so no member can silently drop "
            f"out at one location. A nexus with no rows at all is allowed and "
            f"is handled by the coverage policy."
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


def _to_wide(tidy: pd.DataFrame, formulations: Sequence[str]) -> pd.DataFrame:
    """
    Pivot validated rows into one row per nexus, columns in run order.

    Completeness and duplication have already been checked, so the pivot is
    guaranteed dense and unambiguous, and selecting ``formulations`` from it
    both orders the columns to match the dataset and cannot drop a column.
    """
    order = tidy["nexus_id"].drop_duplicates().tolist()
    wide = tidy.pivot(index="nexus_id", columns="formulation", values="weight")
    return wide.loc[order, list(formulations)].astype(float)


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
    Relabel, validate and normalize the weight groups in a tidy weight frame.

    Pure: plain frames and dicts in, a plain frame out.  No file is read and
    no dataset is touched, so this runs before any Dask graph is built and a
    bad weight file fails before the expensive compute.

    Parameters
    ----------
    weights:
        Tidy weight frame with ``nexus_id``, ``formulation_index`` and
        ``weight`` columns, as ``read_weight_file`` returns it.
    formulation_index_map:
        Legend from weight-file index to formulation name.
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
        The index map does not name exactly the run's formulations, the frame
        is missing a schema column or carries values of the wrong type, a row
        uses an index the map does not define, a group is incomplete or
        duplicated, a weight is negative or non-finite, a group is entirely
        zero, or a group does not sum to 1 and ``normalize`` is False.
    """
    _require_legend_matches_run(formulation_index_map, formulations)
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

    named = _relabel(tidy, formulation_index_map)
    _require_complete_groups(named, formulations)
    _require_valid_weight_values(named)

    wide = _to_wide(named, formulations)
    wide = _apply_sum_rule(wide, normalize=normalize, tolerance=tolerance)

    logger.debug(
        f"Validated {len(wide)} weight group(s) over "
        f"{len(formulations)} formulation(s)."
    )
    return wide


# --------------------------------------------------------------------- #
# Nexus to feature expansion, coverage policy and reporting             #
# --------------------------------------------------------------------- #
@dataclass(frozen=True)
class CoverageReport:
    """
    What a resolution achieved: how much of the run the weight file covered.

    Attributes
    ----------
    total_features:
        Features in the run — the length of the returned weight array's
        ``feature_id`` dimension.
    covered_features:
        Features that drain to a nexus present in the weight file, and so
        carry weights the file supplied.
    uncovered_features:
        Features left on equal weights because their nexus is absent from the
        weight file, or because they drain to no nexus in the crosswalk.
    fraction:
        ``covered_features / total_features``, in ``[0, 1]``.  This is the
        value written to the output NetCDF as provenance.
    used_nexus:
        Weight groups in the file that reached at least one feature.
    unused_nexus:
        Weight groups in the file that reached no feature in this run — the
        normal case when one national weight file is applied to one domain,
        but a large count alongside low coverage points at a wrong crosswalk.
    """

    total_features: int
    covered_features: int
    uncovered_features: int
    fraction: float
    used_nexus: int
    unused_nexus: int

    @property
    def is_complete(self) -> bool:
        """True when every feature in the run carries supplied weights."""
        return self.uncovered_features == 0

    def summary(self) -> str:
        """One-line description, used for the log line and for provenance."""
        return (
            f"weight coverage {self.fraction:.1%} "
            f"({self.covered_features} of {self.total_features} feature(s) "
            f"covered, {self.uncovered_features} uncovered) from "
            f"{self.used_nexus} applied nexus weight group(s)"
            + (f", {self.unused_nexus} unused" if self.unused_nexus else "")
        )


def _nexus_keys(values: Iterable, context: str) -> np.ndarray:
    """
    Reduce nexus identifiers to the integers the hydrofabric's ``toid`` carries.

    ``load_hydrofabric`` strips non-digits from ``toid``, so ``nex-9001``
    becomes ``9001``.  Weight-file nexus ids keep whatever spelling the file
    used.  The reduction itself is
    :func:`teval.identifiers.as_identifiers` — the same one
    ``build_nexus_crosswalk`` applies to the hydrofabric — so the two sides of
    the join cannot normalize differently.  The hazard being guarded is that
    after stripping, a nexus id and a flowpath id are indistinguishable by
    value, and a join against the wrong column would return silently wrong
    weights rather than raise.

    Nothing here is guessed at.  An identifier that carries no digits, or that
    is not integral, raises rather than being reduced to something plausible:
    a misread id would match no nexus, and under the default ``on_missing``
    policy that presents as a coverage shortfall — "your file did not cover
    this domain" — when the truth is that the file was misparsed.
    """
    listed = list(values)
    if not listed:
        return np.empty(0, dtype=np.int64)

    series = pd.Series(listed, dtype=object)

    # Checked here rather than left to as_identifiers, which only sees a bool
    # *dtype*: True is an int in Python and would otherwise key nexus 1.
    booleans = [v for v in listed if isinstance(v, (bool, np.bool_))]
    if booleans:
        raise ValueError(
            f"{context} is not a nexus identifier: "
            f"{_describe([repr(v) for v in booleans])}."
        )

    reduced = as_identifiers(series, context)

    unreadable = reduced.isna()
    if unreadable.any():
        raise ValueError(
            f"{context} carries no digits and cannot be matched against the "
            f"hydrofabric's integer nexus ids: "
            f"{_describe([repr(v) for v in series[unreadable]])}."
        )
    return reduced.to_numpy(dtype=np.int64)


def _as_feature_ids(values: Iterable, context: str) -> np.ndarray:
    """
    Return feature identifiers as a plain int64 array.

    The dtype is forced rather than trusted: an int64 array of feature ids
    compared against a float or object array of the same numbers matches
    nothing, which would read as "no coverage" instead of as a type mismatch.
    """
    array = np.asarray(list(values))
    if array.size == 0:
        return np.empty(0, dtype=np.int64)
    try:
        as_int = array.astype(np.int64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} must be integer feature ids: {exc}") from exc

    if not np.array_equal(as_int.astype(array.dtype, copy=False), array):
        raise ValueError(
            f"{context} must be integer feature ids; some values are not "
            f"integral."
        )
    return as_int


def _crosswalk_arrays(
    nexus_to_features: Mapping,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Flatten the nexus-to-features mapping into aligned feature and nexus arrays.

    Returns
    -------
    (feature_ids, nexus_keys)
        One entry per (feature, nexus) pair, with nexus ids reduced to integer
        keys.  A feature listed twice under the same nexus is harmless and is
        de-duplicated; a feature listed under two different nexuses is an
        error, since its weights would be ambiguous.
    """
    nexus_values = list(nexus_to_features)
    nexus_keys = _nexus_keys(nexus_values, "Crosswalk nexus key")

    features: List[int] = []
    keys: List[int] = []
    for nexus, key in zip(nexus_values, nexus_keys):
        drained_ids = _as_feature_ids(
            nexus_to_features[nexus], f"Crosswalk entry for nexus {nexus}"
        )
        features.extend(int(f) for f in drained_ids)
        keys.extend([int(key)] * len(drained_ids))

    if not features:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

    pairs = pd.DataFrame({"feature_id": features, "nexus_key": keys}).drop_duplicates()
    conflicted = pairs.loc[pairs["feature_id"].duplicated(keep=False), "feature_id"]
    if not conflicted.empty:
        raise ValueError(
            f"Crosswalk assigns feature(s) {_describe(sorted(set(conflicted)))} "
            f"to more than one nexus, so their weights would be ambiguous. A "
            f"flowpath drains to exactly one nexus."
        )
    return (
        pairs["feature_id"].to_numpy(dtype=np.int64),
        pairs["nexus_key"].to_numpy(dtype=np.int64),
    )


def _group_keys(groups: pd.DataFrame) -> np.ndarray:
    """Reduce the validated groups' nexus ids to integer keys, rejecting collisions."""
    keys = _nexus_keys(groups.index, "Weight file nexus_id")
    duplicated = pd.Index(keys).duplicated()
    if duplicated.any():
        collided = sorted(
            {
                str(nexus)
                for nexus, key in zip(groups.index, keys)
                if key in set(keys[duplicated])
            }
        )
        raise ValueError(
            f"Weight file carries nexus ids that reduce to the same nexus "
            f"number: {_describe(collided)}. Their weight groups cannot be "
            f"told apart once matched against the hydrofabric."
        )
    return keys


def _apply_coverage_policy(report: CoverageReport, on_missing: str) -> None:
    """
    Log or abort according to ``on_missing``.

    Called before the weight array is handed back and therefore before any
    Dask graph is built, so the ``error`` path costs one second rather than
    one long compute.
    """
    if on_missing not in ON_MISSING_POLICIES:
        raise ValueError(
            f"Unknown on_missing policy {on_missing!r}; expected one of "
            f"{', '.join(ON_MISSING_POLICIES)}."
        )

    if report.is_complete:
        # Debug rather than info: complete coverage is the uneventful case, and
        # the caller wiring weights into a run emits one summary line per domain
        # naming the file it came from.  Two identical summaries per domain, one
        # of them without the file path, is noise.  Incomplete coverage still
        # warns from here, since that is the resolver's own policy decision.
        logger.debug(f"Weights cover every feature in the run: {report.summary()}.")
        return

    if on_missing == "error":
        raise ValueError(
            f"Incomplete weight coverage and stats.weights.on_missing is "
            f"'error': {report.summary()}. Supply weights for the missing "
            f"nexus, or set on_missing to 'warn' to fall back to equal "
            f"weights for the uncovered features."
        )

    logger.warning(
        f"Incomplete {report.summary()}. The {report.uncovered_features} "
        f"uncovered feature(s) fall back to equal weights, which is the simple "
        f"mean. Set stats.weights.on_missing to 'error' to make this abort "
        f"instead."
    )


def _expand_to_features(
    groups: pd.DataFrame,
    nexus_to_features: Mapping,
    feature_ids: Sequence[int],
    formulations: Sequence[str],
) -> Tuple[xr.DataArray, CoverageReport]:
    """
    Broadcast per-nexus groups onto the run's features and fill the rest.

    Every feature draining to a nexus receives that nexus' group unchanged, so
    the many-to-one relationship needs no arithmetic: a confluence of three
    flowpaths gives all three identical weights.  Features whose nexus supplied
    no weights are filled with ``1 / n_formulations``.
    """
    targets = _as_feature_ids(feature_ids, "feature_ids")
    if targets.size == 0:
        raise ValueError(
            "No feature ids were supplied to expand weights onto; coverage is "
            "undefined for an empty run."
        )
    target_index = pd.Index(targets, name="feature_id")
    if target_index.has_duplicates:
        raise ValueError(
            f"feature_ids carries duplicate value(s): "
            f"{_describe(sorted(set(target_index[target_index.duplicated()])))}. "
            f"The dataset's feature_id coordinate must be unique."
        )

    cross_features, cross_keys = _crosswalk_arrays(nexus_to_features)
    group_keys = _group_keys(groups)

    # Feature -> nexus key, with -1 marking a feature the crosswalk does not
    # place (no hydrofabric row, or a nexus this run does not drain to).
    # -1 never matches a real nexus key, so such features fall through to the
    # uncovered branch without a special case.  The mask is applied before
    # indexing rather than after: a -1 fed to a numpy take wraps to the last
    # row instead of failing, which would hand a feature another nexus' weights.
    from_cross = pd.Index(cross_features).get_indexer(target_index)
    placed = from_cross >= 0
    nexus_of_feature = np.full(target_index.size, -1, dtype=np.int64)
    nexus_of_feature[placed] = cross_keys[from_cross[placed]]

    # Nexus key -> row of the validated group frame; -1 where the file has no
    # group for that nexus.
    group_row = pd.Index(group_keys).get_indexer(nexus_of_feature)
    covered = group_row >= 0

    n_formulations = len(formulations)
    matrix = np.full((target_index.size, n_formulations), 1.0 / n_formulations)
    if covered.any():
        matrix[covered] = groups.to_numpy(dtype=float)[group_row[covered]]

    covered_count = int(covered.sum())
    used_nexus = int(np.unique(group_row[covered]).size)
    report = CoverageReport(
        total_features=int(target_index.size),
        covered_features=covered_count,
        uncovered_features=int(target_index.size) - covered_count,
        fraction=covered_count / int(target_index.size),
        used_nexus=used_nexus,
        unused_nexus=int(len(group_keys)) - used_nexus,
    )

    weights = xr.DataArray(
        matrix,
        dims=("feature_id", "formulation"),
        coords={
            "feature_id": target_index.to_numpy(),
            "formulation": list(formulations),
        },
        name="ensemble_weight",
    )
    return weights, report


def resolve_weights(
    weights: pd.DataFrame,
    formulation_index_map: Mapping[int, str],
    formulations: Sequence[str],
    nexus_to_features: Mapping,
    feature_ids: Sequence[int],
    normalize: bool = False,
    on_missing: str = "warn",
    tolerance: float = SUM_TOLERANCE,
) -> Tuple[xr.DataArray, CoverageReport]:
    """
    Turn a tidy weight frame into a dense weight array over the run's features.

    Validates the groups (see :func:`validate_weight_groups`), expands each
    nexus group across every feature draining to that nexus, fills uncovered
    features with equal weights, and applies the ``on_missing`` coverage
    policy.  Pure: plain frames, dicts and sequences in; a labelled array and
    a report out.  No file is read and no dataset is touched, so both the
    validation errors and the coverage error are raised before any Dask graph
    is built.

    Parameters
    ----------
    weights:
        Tidy weight frame as ``read_weight_file`` returns it.
    formulation_index_map:
        Binding from weight-file index to formulation name.
    formulations:
        The formulation names discovered in the run, in dataset order.
    nexus_to_features:
        Mapping from nexus to the feature ids draining to it, as the
        hydrofabric crosswalk builder returns it.  Nexus ids may be integers
        or prefixed strings; both are reduced the same way.
    feature_ids:
        The run's feature ids, in dataset order.  These become the returned
        array's ``feature_id`` coordinate.
    normalize:
        Divide each group by its own sum instead of requiring it to sum to 1.
    on_missing:
        ``'warn'`` (default) fills uncovered features with equal weights and
        logs the coverage; ``'error'`` raises instead.
    tolerance:
        How far a group's sum may sit from 1.0 when ``normalize`` is False.

    Returns
    -------
    (xr.DataArray, CoverageReport)
        The array is dense over ``(feature_id, formulation)``, coordinates in
        the order supplied, every row summing to 1.  Labelled coordinates mean
        the consumer aligns by name and cannot silently transpose the
        formulation axis.  The report carries the counts and the coverage
        fraction, for logging and for the output NetCDF's provenance.

    Raises
    ------
    ValueError
        Any validation rule fails, the crosswalk or the feature ids are
        malformed, ``on_missing`` is unknown, or coverage is incomplete and
        ``on_missing`` is ``'error'``.
    """
    groups = validate_weight_groups(
        weights,
        formulation_index_map,
        formulations,
        normalize=normalize,
        tolerance=tolerance,
    )
    resolved, report = _expand_to_features(
        groups, nexus_to_features, feature_ids, formulations
    )
    _apply_coverage_policy(report, on_missing)

    logger.debug(
        f"Resolved weights over {report.total_features} feature(s) and "
        f"{len(formulations)} formulation(s); {report.summary()}."
    )
    return resolved, report
