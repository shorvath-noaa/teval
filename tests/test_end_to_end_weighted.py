"""
End-to-end verification of the ensemble weighting feature.

Every layer below this one is already tested in isolation, and
``test_workflow_weights`` joins them at ``load_domain_data`` with the
hydrofabric loader stubbed out.  What no test reaches until here is the run as
a user performs it: a YAML configuration file on disk, T-Route output
directories discovered by name, a real GeoPackage read for its flowpaths and
its gage crosswalk, a weight file joined to that hydrofabric, and the ensemble
NetCDF the pipeline writes -- reopened afterwards and read for its values and
its attributes.  Nothing between ``teval.__main__.main()`` and the file on disk
is stubbed; the only thing kept out of the run is the network, and only because
observations are supplied as a file.

The synthetic domain
--------------------
Three formulations (``cfe``, ``noahowp``, ``lstm``) over four features and six
hourly timesteps, with ``streamflow`` at

    100·member_rank + timestep + 10·feature_rank

so every expectation below is a closed form worked out on paper rather than a
second implementation of the code under test.  Features 101, 102 and 103 drain
to nexus 9001 -- a genuine confluence, so the many-to-one expansion is
exercised through a real GeoPackage -- and 201 drains to nexus 9002.  The
weight file gives 9001 the group 0.5/0.3/0.2 and 9002 the group 0.25/0.75/0.0,
so the member part of the mean is

* 170 at 101, 102 and 103 (``0.5·100 + 0.3·200 + 0.2·300``),
* 175 at 201 (``0.25·100 + 0.75·200 + 0.0·300``),
* 200 wherever weights are absent or equal, which is the plain mean.

The three are deliberately distinct, so a run that ignores the weight file,
applies one nexus' group to the wrong features, or transposes the formulation
axis against the directory scan order cannot pass.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr
import yaml
from shapely.geometry import LineString, Point

from teval import __main__ as teval_main
from teval.weights.provenance import (
    APPLIED_ATTR,
    APPLIED_FALSE,
    APPLIED_TRUE,
    COVERAGE_ATTR,
    FILE_ATTR,
)

#: Domain name, which discovery reads from the run directory suffix.  It
#: doubles as the gage id so the hydrofabric's network layer can name it.
DOMAIN = "01010101"

#: Formulation name to the member offset its file carries.
FORMULATIONS = {"cfe": 100.0, "noahowp": 200.0, "lstm": 300.0}

#: The binding the weight file's integer indices are read through.  It is not
#: the directory scan order, and must not need to be.
INDEX_MAP = {1: "cfe", 2: "noahowp", 3: "lstm"}

#: Feature id to its offset in the closed form, and to the nexus it drains to.
FEATURE_OFFSET = {101: 10.0, 102: 20.0, 103: 30.0, 201: 40.0}
NEXUS_OF = {101: 9001, 102: 9001, 103: 9001, 201: 9002}

#: Weight groups, by nexus, in formulation-index order.
WEIGHT_GROUPS = {9001: [0.5, 0.3, 0.2], 9002: [0.25, 0.75, 0.0]}

N_TIMES = 6
TIMES = pd.date_range("2020-06-01", periods=N_TIMES, freq="h")

#: Member part of the weighted mean, per feature, worked out on paper.
WEIGHTED_MEMBER_PART = {101: 170.0, 102: 170.0, 103: 170.0, 201: 175.0}
#: What a feature with no supplied weights gets: the simple mean.
EQUAL_MEMBER_PART = 200.0
ALL_EQUAL_MEMBER_PARTS = dict.fromkeys(FEATURE_OFFSET, EQUAL_MEMBER_PART)

#: The written NetCDF is cast to float32, so comparisons carry its precision.
FLOAT32_RTOL = 1e-6


# --------------------------------------------------------------------- #
# Keeping a whole-run test from leaking into the rest of the suite      #
# --------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def restore_logging_levels():
    """
    Put the root and ``teval`` logger levels back after each run.

    ``main`` calls ``configure_logging``, which sets both globally.  That is
    correct for a program and unwelcome in a test session, where it would
    outlive this module and change what every later test's ``caplog`` sees.
    """
    root_level = logging.getLogger().level
    teval_level = logging.getLogger("teval").level
    yield
    logging.getLogger().setLevel(root_level)
    logging.getLogger("teval").setLevel(teval_level)


# --------------------------------------------------------------------- #
# The synthetic run, laid out on disk exactly as a real one is          #
# --------------------------------------------------------------------- #
def _member_values(member_offset: float) -> np.ndarray:
    """The closed form for one formulation, shaped ``(time, feature_id)``."""
    return np.array(
        [
            [member_offset + t + FEATURE_OFFSET[f] for f in FEATURE_OFFSET]
            for t in range(N_TIMES)
        ],
        dtype="float64",
    )


def _write_troute_outputs(troute_dir: Path) -> None:
    """One T-Route output directory per formulation, named for discovery."""
    for name, offset in FORMULATIONS.items():
        run_dir = troute_dir / f"{name}_{DOMAIN}_output"
        run_dir.mkdir(parents=True)
        xr.Dataset(
            {"streamflow": (("time", "feature_id"), _member_values(offset))},
            coords={"time": TIMES, "feature_id": list(FEATURE_OFFSET)},
        ).to_netcdf(run_dir / "troute_output.nc", engine="h5netcdf")


def _write_hydrofabric(hydrofabric_dir: Path) -> Path:
    """
    A real GeoPackage, with the prefixed identifiers the loader strips.

    Written rather than stubbed because the crosswalk's central hazard lives in
    this file: ``wb-`` and ``nex-`` are stripped on load, after which a nexus
    number and a feature id are indistinguishable by value.
    """
    hydrofabric_dir.mkdir(parents=True)
    path = hydrofabric_dir / f"hydrofabric_{DOMAIN}.gpkg"

    features = list(FEATURE_OFFSET)
    gpd.GeoDataFrame(
        {
            "id": [f"wb-{f}" for f in features],
            "toid": [f"nex-{NEXUS_OF[f]}" for f in features],
            "hydroseq": list(range(len(features), 0, -1)),
            "order": [1, 1, 2, 3],
            "geometry": [
                LineString([(-100 + i, 40.0), (-100 + i + 0.1, 40.1)])
                for i in range(len(features))
            ],
        },
        crs="EPSG:4326",
    ).to_file(path, layer="flowpaths", driver="GPKG")

    # The gage sits on the tailwater feature, whose nexus carries the second
    # weight group -- so the metrics the run reports are computed from a
    # genuinely weighted series.
    gpd.GeoDataFrame(
        {
            "id": ["wb-201"],
            "toid": ["nex-9002"],
            "hl_uri": [f"gages-{DOMAIN}"],
            "geometry": [Point(-97.0, 40.05)],
        },
        crs="EPSG:4326",
    ).to_file(path, layer="network", driver="GPKG")

    return path


def _write_observations(path: Path) -> None:
    """Six hourly observations at the domain's one gage, no network involved."""
    pd.DataFrame(
        {
            "time": TIMES,
            DOMAIN: [210.0, 214.0, 219.0, 226.0, 222.0, 217.0],
        }
    ).to_csv(path, index=False)


def _write_weight_file(path: Path, nexus_ids) -> Path:
    """The weight groups for *nexus_ids*, in the provisional csv schema."""
    rows = []
    for nexus_id in nexus_ids:
        for index, weight in enumerate(WEIGHT_GROUPS[nexus_id], start=1):
            rows.append(
                {
                    "nexus_id": f"nex-{nexus_id}",
                    "formulation_index": index,
                    "weight": weight,
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


@pytest.fixture
def run_inputs(tmp_path) -> dict:
    """Every input file a run needs, laid out under one temporary root."""
    troute_dir = tmp_path / "troute"
    _write_troute_outputs(troute_dir)
    _write_hydrofabric(tmp_path / "hydrofabric")
    _write_observations(tmp_path / "observations.csv")

    return {
        "root": tmp_path,
        "troute_dir": troute_dir,
        "hydrofabric_dir": tmp_path / "hydrofabric",
        "observations_file": tmp_path / "observations.csv",
        "output_dir": tmp_path / "output",
        "weights_full": _write_weight_file(
            tmp_path / "weights_full.csv", [9001, 9002]
        ),
        "weights_partial": _write_weight_file(
            tmp_path / "weights_partial.csv", [9001]
        ),
    }


def _config_path(run_inputs: dict, weight_file: Path = None, **weight_overrides) -> Path:
    """
    Write the run's YAML configuration, with or without a weights block.

    The configuration goes through the file rather than through ``TevalConfig``
    directly, so the ``stats.weights`` block is parsed from YAML exactly as a
    user's would be -- including the integer keys of
    ``formulation_index_map``, which yaml reads as integers and pydantic must
    accept as such.
    """
    config = {
        "io": {
            "troute_netcdf_dir": str(run_inputs["troute_dir"]),
            "hydrofabric_dir": str(run_inputs["hydrofabric_dir"]),
            "observations_file": str(run_inputs["observations_file"]),
            "output_dir": str(run_inputs["output_dir"]),
            "directory_naming": "suffix",
            "per_domain_output": True,
            "auto_download_usgs": False,
        },
        "system": {"cpu": 1, "logging_level": "DEBUG", "timing": "none"},
        "metrics": {"enabled": True, "variables": ["nse", "pbias"]},
        "viz": {
            "hydrographs": {"enabled": False},
            "skill_maps": {"enabled": False},
            "interactive_map": {"enabled": False},
            "animation": {"enabled": False},
        },
    }
    if weight_file is not None:
        config["stats"] = {
            "weights": {
                "file": str(weight_file),
                "formulation_index_map": dict(INDEX_MAP),
                **weight_overrides,
            }
        }

    path = run_inputs["root"] / "teval_config.yaml"
    with open(path, "w") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    return path


def _run_teval(config_path: Path, monkeypatch) -> None:
    """Run the pipeline through its command-line entry point."""
    monkeypatch.setattr(sys, "argv", ["teval", "-c", str(config_path)])
    teval_main.main()


def _written_ensemble(run_inputs: dict) -> Path:
    """The primary output: the ensemble NetCDF for the synthetic domain."""
    return run_inputs["output_dir"] / DOMAIN / f"{DOMAIN}_ensemble.nc"


def _open_written(run_inputs: dict) -> xr.Dataset:
    """Reopen the written ensemble, loaded so the file handle can be closed."""
    path = _written_ensemble(run_inputs)
    assert path.exists(), f"the run wrote no ensemble NetCDF at {path}"
    with xr.open_dataset(path, engine="h5netcdf") as ds:
        return ds.load()


def _expected_mean(member_part_by_feature: dict, ds: xr.Dataset) -> np.ndarray:
    """
    Rebuild the expected ``(time, feature_id)`` mean from the closed form.

    The whole array is reconstructed, in the order the file itself reports its
    coordinates, so a misalignment on either axis has nowhere to hide.
    """
    features = [int(f) for f in ds.feature_id.values]
    return np.array(
        [
            [member_part_by_feature[f] + t + FEATURE_OFFSET[f] for f in features]
            for t in range(ds.sizes["time"])
        ]
    )


def _mean_of(ds: xr.Dataset) -> np.ndarray:
    """The written mean as a ``(time, feature_id)`` array."""
    return ds["streamflow_mean"].transpose("time", "feature_id").values


# --------------------------------------------------------------------- #
# A weighted run, from configuration file to output file                #
# --------------------------------------------------------------------- #
def test_a_weighted_run_lands_the_weighted_mean_in_streamflow_mean(
    run_inputs, monkeypatch,
):
    """
    The whole point of the feature, asserted on the file the pipeline wrote.

    Discovery, the GeoPackage, the weight file, the resolver, ``build_stats``
    and the NetCDF write are all real here, so this fails if any link in the
    chain is missing rather than only if the arithmetic is wrong.
    """
    _run_teval(_config_path(run_inputs, run_inputs["weights_full"]), monkeypatch)

    ds = _open_written(run_inputs)
    np.testing.assert_allclose(
        _mean_of(ds), _expected_mean(WEIGHTED_MEMBER_PART, ds), rtol=FLOAT32_RTOL
    )


def test_the_written_weighted_mean_is_not_the_plain_mean(run_inputs, monkeypatch):
    """
    Guards the test above against a coincidence.

    If the configured groups happened to reproduce the simple mean, every
    value assertion in this module would pass while the feature did nothing.
    """
    _run_teval(_config_path(run_inputs, run_inputs["weights_full"]), monkeypatch)

    ds = _open_written(run_inputs)
    assert not np.allclose(
        _mean_of(ds), _expected_mean(ALL_EQUAL_MEMBER_PARTS, ds)
    )


def test_only_the_mean_is_weighted_in_the_written_file(run_inputs, monkeypatch):
    """
    Median and the spread band describe the raw members, as documented.

    Read off the same file as the mean, so the caveat the documentation states
    is checked against the product rather than against the source.
    """
    _run_teval(_config_path(run_inputs, run_inputs["weights_full"]), monkeypatch)

    ds = _open_written(run_inputs)
    for variable, member_part in (
        ("streamflow_median", 200.0),
        ("streamflow_min", 100.0),
        ("streamflow_max", 300.0),
    ):
        np.testing.assert_allclose(
            ds[variable].transpose("time", "feature_id").values,
            _expected_mean(dict.fromkeys(FEATURE_OFFSET, member_part), ds),
            rtol=FLOAT32_RTOL,
        )


def test_the_confluence_shares_one_group_and_the_other_nexus_does_not(
    run_inputs, monkeypatch,
):
    """
    The many-to-one expansion survives a real hydrofabric.

    101, 102 and 103 drain to one nexus in the GeoPackage, so their written
    means differ only by the closed form's per-feature offset; 201 carries a
    different group and breaks that pattern.
    """
    _run_teval(_config_path(run_inputs, run_inputs["weights_full"]), monkeypatch)

    mean = _open_written(run_inputs)["streamflow_mean"]
    by_feature = {int(f): mean.sel(feature_id=f).values for f in mean.feature_id.values}

    np.testing.assert_allclose(by_feature[102], by_feature[101] + 10.0, rtol=FLOAT32_RTOL)
    np.testing.assert_allclose(by_feature[103], by_feature[101] + 20.0, rtol=FLOAT32_RTOL)
    assert not np.allclose(by_feature[201], by_feature[101] + 30.0)


# --------------------------------------------------------------------- #
# The unweighted run is the pre-change run                              #
# --------------------------------------------------------------------- #
def _baseline_mean_from_member_files(run_inputs: dict, ds: xr.Dataset) -> np.ndarray:
    """
    The unweighted baseline, recomputed from the member files with numpy.

    This is what the pre-change code produced: ``build_stats`` reduced the
    stacked members with ``combined_ds.mean(dim='formulation')`` and nothing
    else.  Stacking the files here and averaging them with numpy reproduces
    that definition without going through any teval code, so an unweighted run
    agreeing with it agrees with the behaviour that existed before weighting.
    """
    features = [int(f) for f in ds.feature_id.values]
    members = []
    for name in FORMULATIONS:
        path = next((run_inputs["troute_dir"] / f"{name}_{DOMAIN}_output").glob("*.nc"))
        with xr.open_dataset(path, engine="h5netcdf") as member:
            members.append(
                member["streamflow"]
                .sel(feature_id=features)
                .transpose("time", "feature_id")
                .values
            )
    return np.mean(np.stack(members), axis=0)


def test_an_unweighted_run_matches_the_pre_change_baseline(run_inputs, monkeypatch):
    """
    With no weights block, the product is the one the pipeline always made.

    Checked twice over: against the closed form on paper, and against the
    members on disk averaged outside teval entirely.
    """
    _run_teval(_config_path(run_inputs), monkeypatch)

    ds = _open_written(run_inputs)
    np.testing.assert_allclose(
        _mean_of(ds), _expected_mean(ALL_EQUAL_MEMBER_PARTS, ds), rtol=FLOAT32_RTOL
    )
    np.testing.assert_allclose(
        _mean_of(ds), _baseline_mean_from_member_files(run_inputs, ds), rtol=FLOAT32_RTOL
    )


def test_the_unweighted_run_needs_no_weight_machinery(run_inputs, monkeypatch, caplog):
    """
    A run that configures no weights says nothing about them and reads nothing.

    The reader is made to explode, so this fails if the unweighted path reaches
    the weight machinery at all rather than reaching it and discarding the
    result.
    """
    def explode(*args, **kwargs):
        raise AssertionError("the unweighted run read a weight file")

    monkeypatch.setattr("teval.weights.plan.read_weight_file", explode)

    with caplog.at_level(logging.DEBUG, logger="teval"):
        _run_teval(_config_path(run_inputs), monkeypatch)

    assert _written_ensemble(run_inputs).exists()
    assert not [
        record for record in caplog.records
        if record.levelno >= logging.WARNING and "weight" in record.getMessage().lower()
    ]


def test_the_two_runs_differ_only_where_weighting_says_they_should(
    run_inputs, monkeypatch,
):
    """
    The same inputs, run both ways, compared file to file.

    The unweighted run is written second and to the same path, so this also
    pins that a weighted run leaves nothing behind that a later unweighted run
    would inherit.
    """
    _run_teval(_config_path(run_inputs, run_inputs["weights_full"]), monkeypatch)
    weighted = _open_written(run_inputs)

    _run_teval(_config_path(run_inputs), monkeypatch)
    unweighted = _open_written(run_inputs)

    assert set(weighted.data_vars) == set(unweighted.data_vars)
    for variable in ("streamflow_median", "streamflow_min", "streamflow_max"):
        np.testing.assert_array_equal(
            weighted[variable].values, unweighted[variable].values
        )
    assert not np.allclose(
        weighted["streamflow_mean"].values, unweighted["streamflow_mean"].values
    )


# --------------------------------------------------------------------- #
# Partial coverage: the warning and the equal-weight fallback           #
# --------------------------------------------------------------------- #
def test_partial_coverage_warns_and_falls_back_per_feature(
    run_inputs, monkeypatch, caplog,
):
    """
    A file covering one of the two nexus weights part of the domain, and says so.

    The three features at the covered confluence keep the file's group while
    201 falls back to the simple mean, so the fallback is per feature rather
    than an all-or-nothing retreat -- and the warning names the shortfall
    rather than letting the run pass in silence.
    """
    with caplog.at_level(logging.WARNING, logger="teval"):
        _run_teval(
            _config_path(run_inputs, run_inputs["weights_partial"]), monkeypatch
        )

    ds = _open_written(run_inputs)
    expected = {**WEIGHTED_MEMBER_PART, 201: EQUAL_MEMBER_PART}
    np.testing.assert_allclose(
        _mean_of(ds), _expected_mean(expected, ds), rtol=FLOAT32_RTOL
    )

    warnings = [
        record.getMessage()
        for record in caplog.records
        if record.levelno >= logging.WARNING and "uncovered" in record.getMessage()
    ]
    assert len(warnings) == 1
    assert "weight coverage 75.0%" in warnings[0]
    assert "equal weights" in warnings[0]


def test_partial_coverage_under_error_aborts_the_run(run_inputs, monkeypatch):
    """
    The same file with ``on_missing: error`` stops the run instead.

    Asserted from the entry point, and asserted to leave no output: the policy
    exists to prevent a half-weighted product being written at all.
    """
    config = _config_path(
        run_inputs, run_inputs["weights_partial"], on_missing="error"
    )

    with pytest.raises(ValueError, match="on_missing"):
        _run_teval(config, monkeypatch)

    assert not _written_ensemble(run_inputs).exists()


# --------------------------------------------------------------------- #
# Provenance, read back off the written file                            #
# --------------------------------------------------------------------- #
def test_a_weighted_run_records_its_file_and_full_coverage(run_inputs, monkeypatch):
    """The written file names the weight file it was built with, and its reach."""
    _run_teval(_config_path(run_inputs, run_inputs["weights_full"]), monkeypatch)

    attrs = _open_written(run_inputs).attrs
    assert attrs[APPLIED_ATTR] == APPLIED_TRUE
    assert attrs[FILE_ATTR] == str(run_inputs["weights_full"])
    assert attrs[COVERAGE_ATTR] == pytest.approx(1.0)


def test_a_partially_covered_run_records_the_coverage_it_achieved(
    run_inputs, monkeypatch,
):
    """
    Three of four features covered is written as 0.75, not rounded up to 1.0.

    This is the case the attribute exists for: the mean is weighted in part of
    the domain and plain in the rest, and nothing in the values says so.
    """
    _run_teval(_config_path(run_inputs, run_inputs["weights_partial"]), monkeypatch)

    attrs = _open_written(run_inputs).attrs
    assert attrs[APPLIED_ATTR] == APPLIED_TRUE
    assert attrs[FILE_ATTR] == str(run_inputs["weights_partial"])
    assert attrs[COVERAGE_ATTR] == pytest.approx(0.75)


def test_an_unweighted_run_records_that_it_was_not_weighted(run_inputs, monkeypatch):
    """
    The flag is present and negative, and no file or coverage is claimed.

    Present matters as much as negative: a file omitting the attribute
    entirely was produced before weighting existed, which is a different
    statement from "this run applied none".
    """
    _run_teval(_config_path(run_inputs), monkeypatch)

    attrs = _open_written(run_inputs).attrs
    assert attrs[APPLIED_ATTR] == APPLIED_FALSE
    assert FILE_ATTR not in attrs
    assert COVERAGE_ATTR not in attrs


# --------------------------------------------------------------------- #
# The rest of the run still works, and consumes the weighted mean       #
# --------------------------------------------------------------------- #
def _metrics_rows(run_inputs: dict) -> pd.DataFrame:
    """The metrics CSV the run wrote, as a frame."""
    path = run_inputs["output_dir"] / "metrics.csv"
    assert path.exists(), f"the run wrote no metrics CSV at {path}"
    return pd.read_csv(path, dtype={"gage_id": str})


def test_metrics_are_computed_from_the_weighted_mean(run_inputs, monkeypatch):
    """
    Weighting reaches the products derived from the mean, not just the NetCDF.

    The domain's one gage sits on feature 201, whose nexus carries a group that
    is not equal weighting, so the metric scored against the same observations
    must differ between the two runs.  Both runs are performed here rather than
    compared against a literal, since the claim is about the difference.
    """
    _run_teval(_config_path(run_inputs, run_inputs["weights_full"]), monkeypatch)
    weighted = _metrics_rows(run_inputs)

    _run_teval(_config_path(run_inputs), monkeypatch)
    unweighted = _metrics_rows(run_inputs)

    for rows in (weighted, unweighted):
        assert list(rows["source"]) == ["ensemble_mean"]
        assert list(rows["gage_id"]) == [DOMAIN]

    assert weighted["nse"].iloc[0] != pytest.approx(unweighted["nse"].iloc[0])
