"""
teval.io.hydrofabric

Load and prepare NextGen hydrofabric GeoPackages.

Public API
----------
load_hydrofabric(gpkg_path)
    Read a ``.gpkg`` file and return the flowpath GeoDataFrame together with
    the gage crosswalk structures needed downstream.

find_tailwater_feature(gdf_hydro)
    Identify outlet (tailwater) flowpaths in a hydrofabric GeoDataFrame.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import geopandas as gpd
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def load_hydrofabric(
    gpkg_path: Optional[Path],
) -> Tuple[gpd.GeoDataFrame, List[str], Dict, Dict]:
    """
    Load a hydrofabric GeoPackage and build the gage crosswalk.
    Reads the flowpaths and network layers.

    Parameters
    ----------
    gpkg_path:
        Path to the .gpkg file, or None if no hydrofabric is needed
        for this domain (metrics and interactive map will be skipped).

    Returns
    -------
    gdf : gpd.GeoDataFrame
        Flowpath geometries indexed by integer feature_id.  Has columns toid, hydroseq, order,
        geometry, and optionally gage.
    gage_ids : list[str]
        All USGS gage IDs found in the hydrofabric network layer.
    gage_to_fids : dict[str, list[int]]
        Maps each gage ID to the list of upstream feature IDs whose flows
        should be summed at that gage location.
    gage_to_nexus : dict[str, str]
        Maps each gage ID to the nexus toid of the gage row in the
        network layer (used for hydrograph labelling).
    """
    gdf = gpd.GeoDataFrame()
    gage_ids: List[str] = []
    gage_to_fids: Dict = {}
    gage_to_nexus: Dict = {}

    if not gpkg_path:
        return gdf, gage_ids, gage_to_fids, gage_to_nexus

    # Flowpaths
    flowpaths = gpd.read_file(gpkg_path, layer="flowpaths")[
        ["id", "toid", "hydroseq", "order", "geometry"]
    ]
    flowpaths["id"]   = flowpaths["id"].str.replace(r"\D+", "", regex=True).astype(int)
    flowpaths["toid"] = flowpaths["toid"].str.replace(r"\D+", "", regex=True).astype(int)
    flowpaths.set_index("id", inplace=True)

    # Network / gage crosswalk
    network = gpd.read_file(gpkg_path, layer="network")
    gages_net = network[network["hl_uri"].str.startswith("gages-", na=False)].copy()

    if not gages_net.empty:
        gages_net["gage"] = gages_net["hl_uri"].str.replace("gages-", "")

        # Nexus ID
        gage_to_nexus = gages_net.groupby("gage")["toid"].first().to_dict()

        gages_net["id"] = gages_net["id"].str.replace(r"\D+", "", regex=True).astype(int)

        # All upstream feature IDs per gage
        gage_to_fids = (
            gages_net.groupby("gage")["id"]
            .unique()
            .apply(list)
            .to_dict()
        )
        gage_ids = list(gage_to_fids.keys())

        # Assign each gage to the highest stream-order flowpath for map display
        flowpath_gage_df = (
            pd.merge(
                flowpaths["order"].reset_index(),
                gages_net[["id", "gage"]],
                on="id",
            )
            .drop_duplicates()
        )
        flowpath_gage_df = (
            flowpath_gage_df
            .loc[flowpath_gage_df.groupby("gage")["order"].idxmax()][["gage", "id"]]
            .set_index("id")
        )
        flowpaths["gage"] = pd.Series(flowpath_gage_df.to_dict().get("gage"))
    else:
        flowpaths["gage"] = None

    if flowpaths.crs and flowpaths.crs.to_epsg() != 4326:
        gdf = flowpaths.to_crs(epsg=4326)
    else:
        gdf = flowpaths
    return gdf, gage_ids, gage_to_fids, gage_to_nexus


def find_tailwater_feature(gdf_hydro: gpd.GeoDataFrame) -> np.ndarray:
    """
    Identify tailwater flowpaths in a hydrofabric GeoDataFrame.

    Parameters
    ----------
    gdf_hydro:
        GeoDataFrame with integer index (feature IDs) and a toid column.

    Returns
    -------
    np.ndarray
        Array of tailwater feature IDs.
    """
    ids       = gdf_hydro.index
    toids     = gdf_hydro["toid"]
    missing   = ~toids.isin(ids)
    return gdf_hydro.loc[missing].index.values