"""
Tests for ``teval.weights.reader``.

The reader owns the schema and the dtypes and nothing else: it must hand the
resolver exactly what the file said, coerced to known types, or raise.  The
tests below therefore assert on two things — the shape and dtypes of a
successful read, and that every failure mode raises rather than returning a
partial frame that would quietly produce a wrongly weighted product.

Rules the reader deliberately does *not* enforce (duplicates, sign, group
completeness, sums) belong to the resolver and are asserted here to pass
through untouched, so a later change that moves them earlier is caught.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from teval.weights import read_weight_file
from teval.weights.reader import REQUIRED_COLUMNS, SUPPORTED_SUFFIXES


# --------------------------------------------------------------------- #
# Helpers                                                               #
# --------------------------------------------------------------------- #
def write_csv(path: Path, frame: pd.DataFrame) -> Path:
    """Write ``frame`` to ``path`` as CSV and return the path."""
    frame.to_csv(path, index=False)
    return path


def write_parquet(path: Path, frame: pd.DataFrame) -> Path:
    """Write ``frame`` to ``path`` as Parquet and return the path."""
    frame.to_parquet(path, index=False)
    return path


# --------------------------------------------------------------------- #
# A well-formed file parses to the expected frame                       #
# --------------------------------------------------------------------- #
def test_well_formed_csv_parses_to_expected_frame(tmp_path, weight_frame):
    """A well-formed CSV round-trips to the tidy frame it was written from."""
    path = write_csv(tmp_path / "weights.csv", weight_frame)

    result = read_weight_file(path)

    assert_frame_equal(result, weight_frame)


def test_result_schema_is_ordered_and_typed(tmp_path, weight_frame):
    """Columns come back in schema order, on a fresh 0-based index."""
    path = write_csv(tmp_path / "weights.csv", weight_frame)

    result = read_weight_file(path)

    assert list(result.columns) == list(REQUIRED_COLUMNS)
    assert pd.api.types.is_string_dtype(result["nexus_id"])
    assert result["formulation_index"].dtype == "int64"
    assert result["weight"].dtype == "float64"
    assert list(result.index) == list(range(len(weight_frame)))


def test_nex_prefix_is_preserved(tmp_path, weight_frame):
    """The ``nex-`` prefix survives the read.

    Stripping it would make ``nex-9001`` and ``wb-9001`` the same value, so a
    join against a flowpath id would silently succeed on the wrong column.
    """
    path = write_csv(tmp_path / "weights.csv", weight_frame)

    result = read_weight_file(path)

    assert set(result["nexus_id"]) == {"nex-9001", "nex-9002"}


def test_row_order_and_content_are_preserved(tmp_path):
    """Rows come back in file order, unsorted and unaggregated."""
    frame = pd.DataFrame(
        {
            "nexus_id": ["nex-9002", "nex-9001", "nex-9002"],
            "formulation_index": [2, 1, 1],
            "weight": [0.75, 0.5, 0.25],
        }
    )
    path = write_csv(tmp_path / "weights.csv", frame)

    result = read_weight_file(path)

    assert list(result["nexus_id"]) == ["nex-9002", "nex-9001", "nex-9002"]
    assert list(result["formulation_index"]) == [2, 1, 1]
    assert list(result["weight"]) == [0.75, 0.5, 0.25]


def test_extra_columns_are_dropped(tmp_path, weight_frame):
    """A column outside the schema is ignored rather than carried through."""
    frame = weight_frame.assign(provenance=["run-a"] * len(weight_frame))
    path = write_csv(tmp_path / "weights.csv", frame)

    result = read_weight_file(path)

    assert list(result.columns) == list(REQUIRED_COLUMNS)
    assert_frame_equal(result, weight_frame)


def test_header_only_file_returns_empty_typed_frame(tmp_path):
    """A file with only a header is an empty frame, not an error.

    An empty weight file is a coverage outcome for the resolver to report on,
    not a schema violation for the reader to reject.
    """
    path = tmp_path / "weights.csv"
    path.write_text(",".join(REQUIRED_COLUMNS) + "\n")

    result = read_weight_file(path)

    assert result.empty
    assert list(result.columns) == list(REQUIRED_COLUMNS)
    assert result["formulation_index"].dtype == "int64"
    assert result["weight"].dtype == "float64"


# --------------------------------------------------------------------- #
# CSV and Parquet agree                                                 #
# --------------------------------------------------------------------- #
def test_csv_and_parquet_produce_equivalent_frames(tmp_path, weight_frame):
    """The same content in either format reads to the same frame."""
    from_csv = read_weight_file(write_csv(tmp_path / "weights.csv", weight_frame))
    from_parquet = read_weight_file(
        write_parquet(tmp_path / "weights.parquet", weight_frame)
    )

    assert_frame_equal(from_csv, from_parquet)


def test_suffix_dispatch_is_case_insensitive(tmp_path, weight_frame):
    """``.CSV`` dispatches like ``.csv`` — the suffix is lowercased first."""
    path = write_csv(tmp_path / "weights.CSV", weight_frame)

    assert_frame_equal(read_weight_file(path), weight_frame)


@pytest.mark.parametrize("suffix", SUPPORTED_SUFFIXES)
def test_path_may_be_a_string(tmp_path, weight_frame, suffix):
    """A str path is accepted for both formats, not only a Path."""
    path = tmp_path / f"weights{suffix}"
    if suffix == ".parquet":
        write_parquet(path, weight_frame)
    else:
        write_csv(path, weight_frame)

    assert_frame_equal(read_weight_file(str(path)), weight_frame)


# --------------------------------------------------------------------- #
# Missing required columns                                              #
# --------------------------------------------------------------------- #
@pytest.mark.parametrize("missing", REQUIRED_COLUMNS)
def test_missing_required_column_raises(tmp_path, weight_frame, missing):
    """Dropping any required column raises, naming the column and the file."""
    path = write_csv(tmp_path / "weights.csv", weight_frame.drop(columns=[missing]))

    with pytest.raises(ValueError) as excinfo:
        read_weight_file(path)

    message = str(excinfo.value)
    assert missing in message
    assert "weights.csv" in message


def test_missing_column_error_reports_what_was_found(tmp_path, weight_frame):
    """The error names the columns that were present, so a typo is obvious."""
    frame = weight_frame.rename(columns={"nexus_id": "nexus"})
    path = write_csv(tmp_path / "weights.csv", frame)

    with pytest.raises(ValueError, match="nexus"):
        read_weight_file(path)


def test_all_columns_missing_raises(tmp_path):
    """A file sharing none of the schema raises rather than reading as empty."""
    path = write_csv(tmp_path / "weights.csv", pd.DataFrame({"a": [1], "b": [2]}))

    with pytest.raises(ValueError) as excinfo:
        read_weight_file(path)

    for column in REQUIRED_COLUMNS:
        assert column in str(excinfo.value)


# --------------------------------------------------------------------- #
# Dtype coercion                                                        #
# --------------------------------------------------------------------- #
def test_integer_nexus_id_coerces_to_string(tmp_path):
    """A Parquet nexus_id stored as an integer comes back as a string.

    Parquet preserves dtypes, so without coercion this column would arrive as
    int64 and compare unequal to the string ids the crosswalk produces.
    """
    frame = pd.DataFrame(
        {"nexus_id": [9001, 9002], "formulation_index": [1, 1], "weight": [1.0, 1.0]}
    )
    path = write_parquet(tmp_path / "weights.parquet", frame)

    result = read_weight_file(path)

    assert list(result["nexus_id"]) == ["9001", "9002"]
    assert pd.api.types.is_string_dtype(result["nexus_id"])


def test_nexus_id_whitespace_is_stripped(tmp_path):
    """Surrounding whitespace is stripped so ids join as written."""
    frame = pd.DataFrame(
        {
            "nexus_id": ["  nex-9001", "nex-9002  "],
            "formulation_index": [1, 1],
            "weight": [1.0, 1.0],
        }
    )
    path = write_csv(tmp_path / "weights.csv", frame)

    result = read_weight_file(path)

    assert list(result["nexus_id"]) == ["nex-9001", "nex-9002"]


def test_float_valued_formulation_index_coerces_to_int(tmp_path):
    """A whole-numbered float index is coerced, not rejected.

    Parquet and CSV both produce floats for an integer column that shares a
    file with nulls elsewhere, so ``2.0`` must read as ``2``.
    """
    frame = pd.DataFrame(
        {
            "nexus_id": ["nex-9001", "nex-9001"],
            "formulation_index": [1.0, 2.0],
            "weight": [0.4, 0.6],
        }
    )
    path = write_csv(tmp_path / "weights.csv", frame)

    result = read_weight_file(path)

    assert result["formulation_index"].dtype == "int64"
    assert list(result["formulation_index"]) == [1, 2]


def test_string_formulation_index_coerces_to_int(tmp_path):
    """A quoted numeric index still lands as an integer."""
    frame = pd.DataFrame(
        {
            "nexus_id": ["nex-9001", "nex-9001"],
            "formulation_index": ["1", "2"],
            "weight": ["0.4", "0.6"],
        }
    )
    path = write_parquet(tmp_path / "weights.parquet", frame)

    result = read_weight_file(path)

    assert result["formulation_index"].dtype == "int64"
    assert list(result["formulation_index"]) == [1, 2]
    assert result["weight"].dtype == "float64"
    assert list(result["weight"]) == [0.4, 0.6]


def test_integer_weight_coerces_to_float(tmp_path):
    """An integer weight column comes back as float."""
    frame = pd.DataFrame(
        {"nexus_id": ["nex-9001"], "formulation_index": [1], "weight": [1]}
    )
    path = write_parquet(tmp_path / "weights.parquet", frame)

    result = read_weight_file(path)

    assert result["weight"].dtype == "float64"
    assert list(result["weight"]) == [1.0]


def test_non_numeric_formulation_index_raises(tmp_path, weight_frame):
    """A formulation index that is not a number raises, naming the row."""
    frame = weight_frame.astype({"formulation_index": object})
    frame.loc[2, "formulation_index"] = "third"
    path = write_csv(tmp_path / "weights.csv", frame)

    with pytest.raises(ValueError, match="formulation_index"):
        read_weight_file(path)


def test_fractional_formulation_index_raises(tmp_path, weight_frame):
    """A fractional index raises rather than truncating to a wrong member."""
    frame = weight_frame.copy()
    frame["formulation_index"] = frame["formulation_index"].astype(float)
    frame.loc[1, "formulation_index"] = 1.5
    path = write_csv(tmp_path / "weights.csv", frame)

    with pytest.raises(ValueError, match="formulation_index"):
        read_weight_file(path)


def test_non_numeric_weight_raises(tmp_path, weight_frame):
    """A weight that is not a number raises, naming the column."""
    frame = weight_frame.copy()
    frame["weight"] = frame["weight"].astype(object)
    frame.loc[0, "weight"] = "heavy"
    path = write_csv(tmp_path / "weights.csv", frame)

    with pytest.raises(ValueError, match="weight"):
        read_weight_file(path)


@pytest.mark.parametrize("column", REQUIRED_COLUMNS)
def test_null_in_required_column_raises(tmp_path, weight_frame, column):
    """A null anywhere in the schema raises rather than reading as a gap."""
    frame = weight_frame.astype({column: object})
    frame.loc[1, column] = None
    path = write_csv(tmp_path / "weights.csv", frame)

    with pytest.raises(ValueError) as excinfo:
        read_weight_file(path)

    assert column in str(excinfo.value)


def test_blank_nexus_id_raises(tmp_path, weight_frame):
    """A whitespace-only nexus id raises instead of becoming an empty key."""
    frame = weight_frame.copy()
    frame.loc[0, "nexus_id"] = "   "
    path = write_csv(tmp_path / "weights.csv", frame)

    with pytest.raises(ValueError, match="nexus_id"):
        read_weight_file(path)


# --------------------------------------------------------------------- #
# Unreadable or malformed files raise, never partial data               #
# --------------------------------------------------------------------- #
def test_missing_file_raises_file_not_found(tmp_path):
    """An absent path raises FileNotFoundError naming the path."""
    path = tmp_path / "absent.csv"

    with pytest.raises(FileNotFoundError, match="absent.csv"):
        read_weight_file(path)


def test_directory_path_raises_file_not_found(tmp_path):
    """A directory is not a weight file, even when it exists."""
    path = tmp_path / "weights.csv"
    path.mkdir()

    with pytest.raises(FileNotFoundError):
        read_weight_file(path)


def test_unsupported_suffix_raises(tmp_path, weight_frame):
    """An unsupported suffix raises rather than reading as an empty frame.

    ``io.observations`` logs and returns empty on an unknown suffix; here that
    would turn a weighted configuration into a silently unweighted product, so
    the reader raises instead.
    """
    path = tmp_path / "weights.txt"
    weight_frame.to_csv(path, index=False)

    with pytest.raises(ValueError) as excinfo:
        read_weight_file(path)

    message = str(excinfo.value)
    assert ".txt" in message
    for suffix in SUPPORTED_SUFFIXES:
        assert suffix in message


def test_corrupt_parquet_raises_value_error(tmp_path):
    """A file that is not Parquet raises a weight-file error, not a bare one."""
    path = tmp_path / "weights.parquet"
    path.write_bytes(b"this is not a parquet file")

    with pytest.raises(ValueError, match="Could not read weight file"):
        read_weight_file(path)


def test_truncated_parquet_raises_value_error(tmp_path, weight_frame):
    """A half-written Parquet raises rather than yielding the rows it holds."""
    path = write_parquet(tmp_path / "weights.parquet", weight_frame)
    payload = path.read_bytes()
    path.write_bytes(payload[: len(payload) // 2])

    with pytest.raises(ValueError, match="Could not read weight file"):
        read_weight_file(path)


def test_empty_csv_raises_value_error(tmp_path):
    """A zero-byte CSV has no header, so it raises rather than reading empty."""
    path = tmp_path / "weights.csv"
    path.write_bytes(b"")

    with pytest.raises(ValueError):
        read_weight_file(path)


def test_ragged_csv_raises_value_error(tmp_path):
    """A row with more fields than the header raises instead of parsing on."""
    path = tmp_path / "weights.csv"
    path.write_text(
        "nexus_id,formulation_index,weight\n"
        "nex-9001,1,0.5\n"
        "nex-9001,2,0.3,extra,fields\n"
    )

    with pytest.raises(ValueError):
        read_weight_file(path)


# --------------------------------------------------------------------- #
# The reader stops at the schema                                        #
# --------------------------------------------------------------------- #
def test_resolver_rules_are_not_applied_by_the_reader(tmp_path):
    """Duplicates, negatives and a group that does not sum to one pass through.

    These are the resolver's rules.  The reader must hand them over unchanged
    so the resolver reports on what the file actually said.
    """
    frame = pd.DataFrame(
        {
            "nexus_id": ["nex-9001", "nex-9001", "nex-9001"],
            "formulation_index": [1, 1, 2],
            "weight": [-0.5, 2.0, 0.25],
        }
    )
    path = write_csv(tmp_path / "weights.csv", frame)

    result = read_weight_file(path)

    assert len(result) == 3
    assert list(result["weight"]) == [-0.5, 2.0, 0.25]
    assert list(result["formulation_index"]) == [1, 1, 2]
