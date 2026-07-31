"""
teval.identifiers

Reduce NextGen identifiers to the integer form the hydrofabric stores.

Everything that has to be matched against the hydrofabric — the nexus
crosswalk built from its flowpaths, and the weight file joined against that
crosswalk — shares the one reduction here rather than writing its own, so no
two sides of a join can normalize differently.  :func:`as_identifiers` states
what the reduction does and why, and is the single place that rationale is
kept.

Public API
----------
as_identifiers(values, context)
    Reduce an identifier column to numbers, with NA where an entry carries no
    identifier at all.
describe(items)
    Render offending values as a short, truncated list for an error message.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Everything that is not a digit, stripped when reducing a string identifier
# such as "nex-9001" to the integer form load_hydrofabric stores.
_NON_DIGITS = r"\D+"

#: How many offenders an error message names before truncating.  Enough to see
#: a pattern, short enough to stay readable.
MAX_REPORTED = 10


def describe(items) -> str:
    """
    Render a collection as a short, truncated list for an error message.

    Shared by every module that names what it is rejecting -- identifiers,
    formulation names, row positions -- so one bad file reads the same way
    wherever it was caught.
    """
    listed = list(items)[:MAX_REPORTED]
    text = ", ".join(str(item) for item in listed)
    remaining = len(items) - len(listed)
    return f"{text} (and {remaining} more)" if remaining else text or "(none)"


def as_identifiers(
    values: pd.Series,
    context: str,
    required: bool = False,
) -> pd.Series:
    """
    Reduce an identifier column to the integer form the hydrofabric stores.

    ``load_hydrofabric`` strips the ``wb-`` and ``nex-`` prefixes from ``id``
    and ``toid``, so a frame that has been through it carries plain integers
    and ``nex-9001`` is stored as ``9001``.  A frame or a weight file built by
    other means may still hold the prefixed strings, so both are accepted and
    reduced identically here — anything matched against the hydrofabric reduces
    through this function, or one side of a join normalizes differently from
    the other and the join silently finds nothing.

    The prefix is what tells the two kinds of identifier apart, which is why
    callers keep the prefixed spelling for as long as they can and never cross
    a nexus column with a flowpath one: once the prefix is gone, ``nex-123456``
    and ``wb-123456`` are the same number, so a join against the wrong column
    returns silently wrong weights rather than failing.

    The reduction is numeric-first: a value that reads as a number is taken as
    that number, and only a genuinely non-numeric entry is digit-stripped.  The
    order matters.  Digit-stripping first would turn the string ``"9001.0"`` —
    what ``pandas`` produces from a float column, and therefore what an ordinary
    weight file yields when its ``nexus_id`` column lands as float dtype — into
    ``90010`` by swallowing the decimal point, giving a nexus id that matches
    nothing.

    Parameters
    ----------
    values:
        The identifiers to reduce.
    context:
        How to name these identifiers in an error message.  Rendered at the
        start of the sentence, so pass something that reads as a subject, such
        as ``"The 'toid' column"``.
    required:
        If True, an entry that reduces to nothing is an error rather than NA.
        Use it wherever a missing identifier cannot be tolerated: an id that
        matched nothing presents as a coverage shortfall — "your file did not
        cover this domain" — when the truth is that it was misparsed.

    Returns
    -------
    pd.Series
        Float series, positionally aligned with ``values``, carrying NA where
        an identifier is missing or holds no digits at all unless *required*.
        Float rather than int because NA is representable in it; an id above
        2^53 spelled as a string would therefore lose precision, which no
        NextGen identifier is close to reaching.

    Raises
    ------
    ValueError
        The column carries a boolean, or a non-integral number that cannot be
        a hydrofabric identifier, or — with *required* — an entry that reduces
        to nothing.
    """
    # Booleans are checked by value and not only by dtype: an object column
    # holding True is not bool-dtyped, and True is an int in Python, so it
    # would otherwise reduce silently to identifier 1.
    if pd.api.types.is_bool_dtype(values):
        booleans = list(values)
    else:
        raw = values.to_numpy()
        booleans = (
            [v for v in raw if isinstance(v, (bool, np.bool_))]
            if raw.dtype == object
            else []
        )
    if booleans:
        raise ValueError(
            f"{context} carries boolean value(s) {describe(booleans)} and "
            f"cannot hold hydrofabric identifiers."
        )

    numeric = pd.to_numeric(values, errors="coerce")

    # Whatever did not read as a number is a string identifier; strip it down
    # to the digits it carries, exactly as load_hydrofabric does.
    unparsed = numeric.isna() & pd.notna(values)
    if unparsed.any():
        digits = values[unparsed].astype(str).str.replace(_NON_DIGITS, "", regex=True)
        numeric.loc[unparsed] = pd.to_numeric(
            digits.where(digits != ""), errors="coerce"
        )

    fractional = numeric.notna() & (numeric % 1 != 0)
    if fractional.any():
        raise ValueError(
            f"{context} carries non-integer value(s) "
            f"{describe(sorted(set(numeric[fractional].tolist())))}; "
            f"hydrofabric identifiers are integers."
        )

    unreadable = numeric.isna()
    if required and unreadable.any():
        raise ValueError(
            f"{context} carries no digits and cannot be matched against the "
            f"hydrofabric's integer identifiers: "
            f"{describe([repr(v) for v in values[unreadable]])}."
        )
    return numeric
