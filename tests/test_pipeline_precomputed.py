"""
``compute_and_write`` must take the same branch ``workflow`` took.

Whether a domain reuses a pre-computed ensemble is asked in two places: once by
``workflow._process_formulation_files``, which decides whether to build the
statistics, and once here, which decides whether to write them.  They answer
about the same run and so must answer the same way -- if the pipeline believes
the statistics were pre-computed while the workflow has just built them, the
full-domain NetCDF is never written and ``_full_nc_path`` names a file that
does not exist.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from teval import pipeline, workflow
from teval.config import TevalConfig


@pytest.fixture
def stats_ds(combined_ds):
    """A lazy statistics dataset in the shape ``build_stats`` returns one."""
    ds_mean = combined_ds.mean(dim="formulation", keep_attrs=True)
    return ds_mean.rename({v: f"{v}_mean" for v in ds_mean.data_vars})


@pytest.fixture
def config(tmp_path):
    """A configuration that writes its output under ``tmp_path``."""
    config = TevalConfig()
    config.io.output_dir = tmp_path / "out"
    config.io.per_domain_output = True
    return config


def _run(domain_data, ensemble_file, config):
    domain_dict = {
        "formulations": {"raw_files": {}, "ensemble_file": ensemble_file},
    }
    return pipeline.compute_and_write("dom", domain_data, domain_dict, config)


def test_statistics_built_here_are_written_here(stats_ds, config):
    """A domain whose statistics this run built must have them written to disk."""
    domain_data = {"formulations": {"combined": stats_ds, "ensemble_members": None}}

    _run(domain_data, None, config)

    written = domain_data["formulations"]["_full_nc_path"]
    assert written is not None and written.exists(), (
        f"the statistics this run built were not written; _full_nc_path is "
        f"{written}"
    )


def test_a_named_but_absent_ensemble_file_stops_the_write(stats_ds, config):
    """
    The pipeline refuses the same configuration the workflow refuses.

    This is the case the two modules disagreed about: ``pipeline`` reported it
    pre-computed and skipped the write while ``workflow`` built the statistics
    from raw.  Both now raise, so neither can act on a file that is not there.
    """
    domain_data = {"formulations": {"combined": stats_ds, "ensemble_members": None}}

    with pytest.raises(FileNotFoundError, match="never_written.nc"):
        _run(domain_data, Path("/nonexistent/never_written.nc"), config)


def test_an_existing_pre_computed_ensemble_is_not_rewritten(
    tmp_path, stats_ds, config
):
    """The branch still holds for a real pre-computed ensemble: reuse, no write."""
    ensemble_file = tmp_path / "precomputed.nc"
    stats_ds.to_netcdf(ensemble_file, engine="h5netcdf")

    domain_data = {"formulations": {"combined": stats_ds, "ensemble_members": None}}

    _run(domain_data, ensemble_file, config)

    assert domain_data["formulations"]["_full_nc_path"] == ensemble_file
    assert not (config.io.output_dir / "dom" / "dom_ensemble.nc").exists()


def test_both_sites_ask_the_same_question(tmp_path):
    """
    The case the divergence turned on now has one answer: refusal.

    ``.get("ensemble_file") is None`` called a named-but-absent file a reuse
    where ``reuses_precomputed_ensemble`` called it a rebuild.  Neither reading
    survives -- the configuration is refused before either branch is taken.
    """
    formulations = {"raw_files": {"a": Path("a.nc")}, "ensemble_file": tmp_path / "absent.nc"}

    with pytest.raises(FileNotFoundError, match="absent.nc"):
        workflow.reuses_precomputed_ensemble(formulations)
