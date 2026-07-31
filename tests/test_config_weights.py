"""
Tests for the optional ``stats.weights`` configuration block.

The block is cohesive — a weight file path is meaningless without the legend
binding formulation indices to names — so both are required whenever the block
is present, while the block itself stays absent by default.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from teval.config import (
    StatsConfig,
    TevalConfig,
    WeightsConfig,
    generate_config_help,
    generate_default_config,
)


# --------------------------------------------------------------------- #
# Absence is the default                                                #
# --------------------------------------------------------------------- #
def test_weights_absent_by_default():
    """A config with no weights block validates and carries weights=None."""
    assert TevalConfig().stats.weights is None
    assert StatsConfig().weights is None


def test_config_without_weights_block_validates(tmp_path):
    """An existing YAML that never heard of weights still loads unchanged."""
    path = tmp_path / "teval_config.yaml"
    path.write_text(
        yaml.dump({"stats": {"quantiles": [0.1, 0.9], "small_domain_threshold": 4}})
    )

    config = TevalConfig.from_yaml(path)

    assert config.stats.quantiles == [0.1, 0.9]
    assert config.stats.small_domain_threshold == 4
    assert config.stats.weights is None


def test_explicit_null_weights_validates():
    """Writing the key out as null is equivalent to omitting it."""
    assert StatsConfig(weights=None).weights is None


def test_generate_default_config_succeeds(tmp_path):
    """The default config generator still round-trips through YAML."""
    path = tmp_path / "default.yaml"

    generate_default_config(str(path))

    raw = yaml.safe_load(path.read_text())
    assert "weights" in raw["stats"]
    assert raw["stats"]["weights"] is None
    assert TevalConfig(**raw).stats.weights is None


def test_config_help_documents_the_weights_field():
    """generate_config_help picks the weights field up under the stats section."""
    help_text = generate_config_help()

    assert "• weights (Default: None)" in help_text
    assert "spatially varying ensemble weights" in help_text


# --------------------------------------------------------------------- #
# Field defaults and coercion                                           #
# --------------------------------------------------------------------- #
def test_weights_block_defaults(formulation_index_map):
    """on_missing defaults to warn and normalize to false."""
    weights = WeightsConfig(
        file="weights.csv", formulation_index_map=formulation_index_map
    )

    assert weights.on_missing == "warn"
    assert weights.normalize is False
    assert weights.file == Path("weights.csv")


def test_weights_block_from_yaml(tmp_path, formulation_index_map):
    """A full weights block loads through the root config."""
    path = tmp_path / "teval_config.yaml"
    path.write_text(
        yaml.dump(
            {
                "stats": {
                    "weights": {
                        "file": "data/weights.parquet",
                        "formulation_index_map": formulation_index_map,
                        "on_missing": "error",
                        "normalize": True,
                    }
                }
            }
        )
    )

    weights = TevalConfig.from_yaml(path).stats.weights

    assert weights.file == Path("data/weights.parquet")
    assert weights.formulation_index_map == formulation_index_map
    assert weights.on_missing == "error"
    assert weights.normalize is True


def test_formulation_index_map_keys_coerce_to_int(formulation_index_map):
    """YAML quoting the indices must not change the binding."""
    weights = WeightsConfig(
        file="weights.csv",
        formulation_index_map={str(k): v for k, v in formulation_index_map.items()},
    )

    assert weights.formulation_index_map == formulation_index_map


# --------------------------------------------------------------------- #
# Rejected configurations                                               #
# --------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "missing_field", ["file", "formulation_index_map"]
)
def test_partial_weights_block_rejected(missing_field, formulation_index_map):
    """Neither half of the block is useful alone, so both are required."""
    payload = {
        "file": "weights.csv",
        "formulation_index_map": formulation_index_map,
    }
    payload.pop(missing_field)

    with pytest.raises(ValidationError) as excinfo:
        WeightsConfig(**payload)

    assert missing_field in str(excinfo.value)


def test_unknown_on_missing_value_rejected(formulation_index_map):
    """on_missing is closed over warn and error."""
    with pytest.raises(ValidationError):
        WeightsConfig(
            file="weights.csv",
            formulation_index_map=formulation_index_map,
            on_missing="ignore",
        )


def test_zero_based_formulation_index_map_rejected():
    """Indices are 1-based, so a 0 index is a config error, not an off-by-one."""
    with pytest.raises(ValidationError, match="1-based"):
        WeightsConfig(file="weights.csv", formulation_index_map={0: "formA"})


def test_negative_formulation_index_rejected():
    """1-based means positive; a negative key is wrong whatever the run holds."""
    with pytest.raises(ValidationError, match="1-based"):
        WeightsConfig(
            file="weights.csv",
            formulation_index_map={-3: "formA", 1: "formB"},
        )


def test_empty_formulation_index_map_rejected():
    """
    A legend binding nothing is wrong whatever the run holds, so it fails here.

    The resolver would catch it too, as an unmapped formulation — but only
    per-domain, after discovery and hydrofabric loading.  Nothing about the
    run is needed to know an empty map is useless, so the run stops at config
    load instead.
    """
    with pytest.raises(ValidationError, match="empty"):
        WeightsConfig(file="weights.csv", formulation_index_map={})


def test_maps_only_the_run_can_judge_are_left_to_the_resolver():
    """
    Config validates what it can decide alone and nothing that depends on the run.

    Whether a legend binds the right names is a comparison against the
    formulations the run discovered, which configuration cannot see.  A map
    spending two indices on one name leaves some other formulation unbound —
    but which one, and whether it matters, is the resolver's set comparison to
    make (see ``test_weights_resolve.py``); rejecting the map here would only
    mean two different messages for one mistake.
    """
    index_map = {1: "formA", 2: "formA"}

    weights = WeightsConfig(file="weights.csv", formulation_index_map=index_map)

    assert weights.formulation_index_map == index_map
