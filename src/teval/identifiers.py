"""
teval.identifiers

The one reduction of NextGen identifiers, and the one way errors name them.

Public API
----------
as_identifiers(values, context, required=False)
    Reduce an identifier column to numbers, with NA where an entry carries no
    identifier at all.  Its docstring holds the rationale every caller relies
    on, including why they all have to come through here.
describe(items)
    Render offending values as a short, truncated list for an error message.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Stripped from a string identifier like "nex-9001".
_NON_DIGITS = r"\D+"

#: How many offenders an error message names before truncating.
MAX_REPORTED = 10


def describe(items) -> str:
    """
    Render a collection as a short, truncated list for an error message.

    Used by every module that names what it is rejecting, so one bad file
    reads the same way wherever it was caught.
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

    ``load_hydrofabric`` reduces through this function, so a frame that has
    been through it stores ``nex-9001`` as ``9001``, while a weight file or a
    frame built by other means may still hold the prefixed string.
    Both are accepted and reduced identically, and every side of a join comes
    through here: two sides normalizing differently is a join that silently
    finds nothing.

    The prefix is what tells a nexus from a flowpath, which is why callers keep
    it for as long as they can and never cross the two columns — once it is
    gone ``nex-123456`` and ``wb-123456`` are the same number, so a join
    against the wrong one returns silently wrong weights rather than failing.

    The reduction is numeric-first, and that order matters: digit-stripping
    first would swallow the decimal point in ``"9001.0"`` — which is what
    ``pandas`` renders from a float ``nexus_id`` column, and so how an ordinary
    weight file arrives — and give ``90010``, an id that matches nothing.

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
        Float because NA is representable in it, so an id above 2^53 spelled
        as a string loses precision — no NextGen identifier comes close.

    Raises
    ------
    ValueError
        The column carries a boolean, or a non-integral number that cannot be
        a hydrofabric identifier, or — with *required* — an entry that reduces
        to nothing.
    """
    # By value and not only by dtype: an object column holding True is not
    # bool-dtyped, and True is an int in Python, so it would key identifier 1.
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

    # Whatever did not read as a number is a string identifier; strip it to its
    # digits, as load_hydrofabric does.
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

    if required:
        unreadable = numeric.isna()
        if unreadable.any():
            raise ValueError(
                f"{context} carries no digits and cannot be matched against "
                f"the hydrofabric's integer identifiers: "
                f"{describe([repr(v) for v in values[unreadable]])}."
            )
    return numeric
