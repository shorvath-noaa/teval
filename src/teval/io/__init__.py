"""
teval.io

Input/output helpers for the teval pipeline.

Submodules
----------
discovery     — scan T-Route output directories, build the domain map
hydrofabric   — load NextGen GeoPackages, build gage crosswalks
observations  — load USGS streamflow observations from file or API
"""

from teval.io.discovery import (
    parse_run_directory,
    discover_formulation_files,
    initialize_domains,
)
from teval.io.hydrofabric import (
    load_hydrofabric,
    build_nexus_crosswalk,
    find_tailwater_feature,
)
from teval.io.observations import fetch_observations

__all__ = [
    "parse_run_directory",
    "discover_formulation_files",
    "initialize_domains",
    "load_hydrofabric",
    "build_nexus_crosswalk",
    "find_tailwater_feature",
    "fetch_observations",
]