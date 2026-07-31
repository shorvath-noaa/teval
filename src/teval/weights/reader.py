"""
teval.weights.reader

Read a weight file into a tidy DataFrame with a validated schema and dtypes.

The provisional weight file is a tidy table with three columns::

    nexus_id, formulation_index, weight

``nexus_id`` is a string keeping its ``nex-`` prefix, ``formulation_index`` is
a 1-based integer and ``weight`` is a float.  CSV and Parquet are both
accepted, matching the convention used for ``observations_file``.

This module holds no domain knowledge and performs no resolution: it does not
know which formulations exist, which features drain to which nexus, or what a
group of weights must sum to.  Those rules live in ``teval.weights.resolve``.
Only the schema and the dtypes are enforced here, so this is the single module
that changes when the real weight format is defined.

The format is provisional.  The process that will generate these weights does
not yet exist and the format is expected to change.

Public API
----------
read_weight_file(file_path)
    Return a tidy DataFrame of weights with columns nexus_id (str),
    formulation_index (int) and weight (float).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

import pandas as pd

logger = logging.getLogger(__name__)

#: Columns a weight file must carry.  Any other column is dropped.
REQUIRED_COLUMNS = ("nexus_id", "formulation_index", "weight")

#: File suffixes this reader can dispatch on.
SUPPORTED_SUFFIXES = (".csv", ".parquet")

# How many offending row positions to name in an error message before
# truncating.  Enough to see a pattern, short enough to stay readable.
_MAX_REPORTED_ROWS = 10


def _describe_rows(positions) -> str:
    """Render 0-based row positions as a short, truncated list for an error."""
    listed = [int(p) for p in positions[:_MAX_REPORTED_ROWS]]
    text = ", ".join(str(p) for p in listed)
    remaining = len(positions) - len(listed)
    return f"{text} (and {remaining} more)" if remaining else text


def _load_frame(file_path: Path) -> pd.DataFrame:
    """
    Dispatch on the file suffix and return the raw, uncoerced DataFrame.

    Read failures are re-raised as ValueError naming the file, so a truncated
    or otherwise malformed file surfaces as a weight-file problem rather than
    as a bare parser error from pandas or pyarrow.
    """
    suffix = file_path.suffix.lower()

    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(
            f"Unsupported weight file format '{file_path.suffix}' for "
            f"{file_path}. Supported formats: "
            f"{', '.join(SUPPORTED_SUFFIXES)}."
        )

    logger.debug(f"Loading weights from {suffix} file: {file_path}")

    try:
        if suffix == ".parquet":
            return pd.read_parquet(file_path)
        return pd.read_csv(file_path)
    except Exception as exc:
        raise ValueError(f"Could not read weight file {file_path}: {exc}") from exc


def _require_columns(df: pd.DataFrame, file_path: Path) -> None:
    """Raise if any required column is absent, naming what was found instead."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Weight file {file_path} is missing required column(s): "
            f"{', '.join(missing)}. Found columns: "
            f"{', '.join(str(c) for c in df.columns) or '(none)'}. "
            f"A weight file must carry {', '.join(REQUIRED_COLUMNS)}."
        )


def _require_no_nulls(df: pd.DataFrame, file_path: Path) -> None:
    """Raise if any required column carries a null, naming the rows."""
    for column in REQUIRED_COLUMNS:
        null_positions = df.index[df[column].isna()].tolist()
        if null_positions:
            raise ValueError(
                f"Weight file {file_path} has missing values in column "
                f"'{column}' at row(s) {_describe_rows(null_positions)}. "
                f"Every row must carry all of {', '.join(REQUIRED_COLUMNS)}."
            )


def _coerce_nexus_id(values: pd.Series, file_path: Path) -> pd.Series:
    """
    Coerce nexus identifiers to stripped strings.

    The ``nex-`` prefix is deliberately preserved rather than stripped to an
    integer: after prefix stripping ``nex-123456`` and ``wb-123456`` are the
    same value, so a nexus identifier would become indistinguishable from a
    flowpath identifier and a join on the wrong column would fail silently.

    A column of unprefixed ids lands as a numeric dtype, and ``astype(str)``
    then spells it ``"9001.0"``.  That is left as it is rather than special-
    cased here: reducing an identifier for the join is
    :func:`teval.identifiers.as_identifiers`' job, and it reads such a value as
    a number before considering its digits, so ``"9001.0"`` matches nexus 9001.
    """
    nexus_id = values.astype(str).str.strip()

    blank_positions = nexus_id.index[nexus_id == ""].tolist()
    if blank_positions:
        raise ValueError(
            f"Weight file {file_path} has blank 'nexus_id' values at row(s) "
            f"{_describe_rows(blank_positions)}."
        )
    return nexus_id


def _coerce_formulation_index(values: pd.Series, file_path: Path) -> pd.Series:
    """Coerce formulation indices to integers, rejecting non-integral values."""
    numeric = pd.to_numeric(values, errors="coerce")

    bad_positions = numeric.index[numeric.isna()].tolist()
    if bad_positions:
        raise ValueError(
            f"Weight file {file_path} has non-numeric 'formulation_index' "
            f"values at row(s) {_describe_rows(bad_positions)}: "
            f"{sorted(set(values.loc[bad_positions].astype(str)))}. "
            f"'formulation_index' must be a 1-based integer."
        )

    fractional_positions = numeric.index[numeric % 1 != 0].tolist()
    if fractional_positions:
        raise ValueError(
            f"Weight file {file_path} has non-integer 'formulation_index' "
            f"values at row(s) {_describe_rows(fractional_positions)}: "
            f"{sorted(set(numeric.loc[fractional_positions]))}. "
            f"'formulation_index' must be a 1-based integer."
        )
    return numeric.astype(int)


def _coerce_weight(values: pd.Series, file_path: Path) -> pd.Series:
    """Coerce weights to floats, rejecting values that are not numeric."""
    numeric = pd.to_numeric(values, errors="coerce")

    bad_positions = numeric.index[numeric.isna()].tolist()
    if bad_positions:
        raise ValueError(
            f"Weight file {file_path} has non-numeric 'weight' values at "
            f"row(s) {_describe_rows(bad_positions)}: "
            f"{sorted(set(values.loc[bad_positions].astype(str)))}. "
            f"'weight' must be a float."
        )
    return numeric.astype(float)


def read_weight_file(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Read a weight file into a tidy DataFrame with validated dtypes.

    Parameters
    ----------
    file_path:
        Path to the weight file.  ``.csv`` and ``.parquet`` are supported.

    Returns
    -------
    pd.DataFrame
        Columns ``nexus_id`` (str, ``nex-`` prefix retained),
        ``formulation_index`` (int) and ``weight`` (float), in that order,
        with a fresh 0-based index.  Rows are returned in file order; any
        additional column in the file is dropped.  A file carrying only a
        header returns an empty frame with those dtypes.

    Raises
    ------
    FileNotFoundError
        The file does not exist, or is not a regular file.
    ValueError
        The suffix is not supported, the file cannot be parsed, a required
        column is absent, or a value cannot be coerced to its column's type.
        Nothing partial is returned: any problem raises.

    Notes
    -----
    Only the schema is checked here.  Duplicate rows, group completeness,
    sign, and sums are the resolver's concern and are deliberately left
    untouched, so the resolver sees exactly what the file said.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Weight file not found: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Weight file is not a regular file: {path}")

    raw = _load_frame(path)

    _require_columns(raw, path)

    df = raw[list(REQUIRED_COLUMNS)].reset_index(drop=True)

    dropped = [str(c) for c in raw.columns if c not in REQUIRED_COLUMNS]
    if dropped:
        logger.debug(
            f"Ignoring column(s) not in the weight file schema: "
            f"{', '.join(dropped)}"
        )

    _require_no_nulls(df, path)

    df["nexus_id"] = _coerce_nexus_id(df["nexus_id"], path)
    df["formulation_index"] = _coerce_formulation_index(df["formulation_index"], path)
    df["weight"] = _coerce_weight(df["weight"], path)

    if df.empty:
        logger.warning(f"Weight file {path} contains no rows.")
    else:
        logger.debug(
            f"Weights loaded: {len(df)} rows, "
            f"{df['nexus_id'].nunique()} nexus, "
            f"{df['formulation_index'].nunique()} formulation indices"
        )
    return df
