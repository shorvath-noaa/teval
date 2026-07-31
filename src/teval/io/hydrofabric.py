"""
teval.io.hydrofabric

Load and prepare NextGen hydrofabric GeoPackages.

Public API
----------
load_hydrofabric(gpkg_path)
    Read a ``.gpkg`` file and return the flowpath GeoDataFrame together with
    the gage crosswalk structures needed downstream.

build_nexus_crosswalk(gdf_hydro)
    Map each nexus to the feature ids draining to it, for weighting.

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

# Everything that is not a digit, stripped when reducing an identifier such as
# "nex-9001" to the integer form load_hydrofabric stores.
_NON_DIGITS = r"\D+"


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


def _as_identifiers(values: pd.Series, context: str) -> pd.Series:
    """
    Reduce an identifier column to the integer form the hydrofabric stores.

    ``load_hydrofabric`` already strips non-digits from ``id`` and ``toid``, so
    a frame that has been through it carries plain integers.  A frame built by
    other means may still hold the prefixed strings (``nex-9001``), so both are
    accepted and reduced identically — the two must not normalize differently
    or a flowpath would be filed under the wrong nexus.

    Numeric values are read as numbers first and only genuinely non-numeric
    entries are digit-stripped, so a float ``9001.0`` sitting in an object
    column cannot be read as ``90010`` by having its decimal point removed.

    Returns
    -------
    pd.Series
        Float series, positionally aligned with ``values``, carrying NA where
        an identifier is missing or holds no digits at all.

    Raises
    ------
    ValueError
        The column is boolean, or carries a non-integral number that cannot be
        a hydrofabric identifier.
    """
    if pd.api.types.is_bool_dtype(values):
        raise ValueError(
            f"The {context} column is boolean and cannot hold hydrofabric "
            f"identifiers."
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
        offenders = sorted(set(numeric[fractional].tolist()))[:10]
        raise ValueError(
            f"The {context} column carries non-integer value(s) {offenders}; "
            f"hydrofabric identifiers are integers."
        )
    return numeric


def build_nexus_crosswalk(
    gdf_hydro: Optional[gpd.GeoDataFrame],
) -> Dict[int, List[int]]:
    """
    Map each nexus to the feature ids draining to it.

    Ensemble weights are supplied per nexus while the ensemble dataset is
    indexed by ``feature_id``, so applying them needs this crosswalk.  The
    relationship is many-to-one: every flowpath at a confluence drains to the
    same nexus and therefore shares its weights.

    The mapping is derived entirely from the ``toid`` column of the flowpaths
    frame that has already been loaded, so this reads no file and adds no
    GeoPackage access to a run.

    The nexus keys come from ``toid`` and the feature ids from the frame's
    index, and the two are never crossed.  That separation is the point of the
    function: ``load_hydrofabric`` strips the ``wb-`` and ``nex-`` prefixes, so
    afterwards a nexus number and a flowpath id are indistinguishable by value
    and a mapping built from the wrong column would return silently wrong
    weights rather than fail.

    Parameters
    ----------
    gdf_hydro:
        Flowpaths frame as ``load_hydrofabric`` returns it: indexed by integer
        feature id with a ``toid`` column naming the downstream nexus.  ``None``
        or an empty frame means no hydrofabric was configured for this domain.

    Returns
    -------
    dict[int, list[int]]
        Nexus id to the feature ids draining to it, both as plain integers.
        Nexus keys follow first appearance in the frame and each list follows
        frame order, so the result is deterministic.  A nexus appears only if
        at least one flowpath drains to it.  An absent or empty hydrofabric
        gives an empty dict, which resolves to no coverage rather than to an
        error — the caller decides whether missing weights matter.

    Raises
    ------
    ValueError
        A non-empty frame has no ``toid`` column, an identifier column holds a
        non-integral value, or a feature id is missing.

    Notes
    -----
    Flowpaths whose ``toid`` is missing are dropped with a warning: they cannot
    be placed at a nexus, and they surface downstream as uncovered features
    under the weight coverage policy.  A feature that drains to two different
    nexuses is left in both groups for the resolver to reject, so that rule
    lives in one place.
    """
    if gdf_hydro is None or len(gdf_hydro) == 0:
        logger.debug(
            "No hydrofabric flowpaths available; nexus crosswalk is empty."
        )
        return {}

    if "toid" not in gdf_hydro.columns:
        found = ", ".join(str(c) for c in gdf_hydro.columns) or "(none)"
        raise ValueError(
            f"Flowpaths frame has no 'toid' column, so the nexus each flowpath "
            f"drains to is unknown and no crosswalk can be built. Found "
            f"column(s): {found}."
        )

    # Both sides are pulled onto a fresh positional index so the id taken from
    # the frame's index and the toid taken from the column stay row-aligned.
    features = _as_identifiers(pd.Series(gdf_hydro.index.to_numpy()), "flowpath id")
    nexus = _as_identifiers(pd.Series(gdf_hydro["toid"].to_numpy()), "toid")

    if features.isna().any():
        raise ValueError(
            f"{int(features.isna().sum())} flowpath(s) carry no usable feature "
            f"id in the frame's index, so they cannot be crosswalked to a nexus."
        )

    unplaced = int(nexus.isna().sum())
    if unplaced:
        logger.warning(
            f"{unplaced} flowpath(s) carry no usable 'toid' and are left out of "
            f"the nexus crosswalk; they will count as uncovered if weights are "
            f"applied."
        )

    pairs = (
        pd.DataFrame({"feature_id": features, "nexus_id": nexus})
        .dropna()
        .astype({"feature_id": "int64", "nexus_id": "int64"})
        .drop_duplicates()
    )

    crosswalk = {
        int(nexus_id): [int(feature_id) for feature_id in group]
        for nexus_id, group in pairs.groupby("nexus_id", sort=False)["feature_id"]
    }

    logger.debug(
        f"Built nexus crosswalk: {len(crosswalk)} nexus over "
        f"{len(pairs)} flowpath(s)."
    )
    return crosswalk


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