"""
teval/ensemble_methods/attributes.py

Loads basin/catchment attributes from the NextGen hydrofabric GeoPackage and
joins them to feature_ids so that spatially-varying ensemble combination
methods can condition weights on physical basin characteristics.

Attribute source
----------------
The NextGen hydrofabric stores catchment-level attributes across two layers:

    ``divide-attributes``
        Model parameters and terrain/soil descriptors aggregated per catchment
        (divide).  Keyed by ``divide_id`` (e.g. ``cat-12345``).

    ``divides``
        Geometric and topological properties of each catchment, including
        drainage area.  Also keyed by ``divide_id``.

Attributes are pulled from both layers and merged on ``divide_id``.

Join strategy
-------------
The join key between flowpath IDs and divide IDs is the NUMERIC part of
their prefixed identifiers:

    flowpath-attributes.id  = "wb-12345"   →  numeric_id = 12345
    divide-attributes.divide_id = "cat-12345"  →  numeric_id = 12345

This approach is robust to hydrofabric versions where ``flowpath-attributes``
may or may not contain a ``divide_id`` crosswalk column.

# NOTE / TODO: This module currently uses LOCAL catchment attributes — the
# attributes of the catchment that the flowpath drains directly.  For routed
# network applications the scientifically preferred quantity is the
# UPSTREAM-ACCUMULATED attribute — the area-weighted mean of all contributing
# catchments upstream of the feature_id.  Upstream accumulation requires a
# network traversal (similar to what subset_to_domains.py does) and is left
# as a future improvement.  The current approach is a reasonable first
# approximation, especially for small headwater domains where the local and
# accumulated values are nearly identical.

Default attributes used
-----------------------
The following columns are pulled from the hydrofabric layers.  Not all may
be present in every version — missing columns are silently dropped with a
warning, and the fallback equal-weights path is used for any feature_id with
no attribute data.

From ``divide-attributes``:

  mean.slope                        : mean catchment slope (degrees, 30m DEM)
  mean.elevation                    : mean catchment elevation (m, 30m DEM)
  mean.impervious                   : mean impervious fraction (%, 2021 NLCD)
  mean.refkdt                       : Noah-MP surface runoff parameter;
                                      controls infiltration vs. runoff
                                      partitioning — higher = more infiltration
  mean.mfsno                        : snowmelt m parameter; controls snowmelt
                                      dynamics in Noah-MP — key discriminator
                                      for seasonal weight differences in
                                      snow-affected catchments
  mean.smcmax_soil_layers_stag=1    : top-layer soil porosity (m3/m3)
  mean.smcwlt_soil_layers_stag=1    : top-layer wilting point (m3/m3)
  geom_mean.dksat_soil_layers_stag=1: top-layer saturated hydraulic
                                      conductivity (mm/h); proxy for soil
                                      permeability
  geom_mean.psisat_soil_layers_stag=1: top-layer saturated soil matric
                                      potential (kPa); complements dksat by
                                      capturing how strongly the soil matrix
                                      holds water
  dist_4.twi                        : topographic wetness index, 4th quartile
                                      (75th percentile) of TWI distribution
                                      within each catchment — emphasises the
                                      wetter/flatter portions
  mean.Coeff                        : groundwater coefficient (m3/s); controls
                                      baseflow response — discriminates
                                      formulations with different subsurface
                                      parameterizations (TOPMODEL vs SAC-SMA)
  mean.Zmax                         : baseflow bucket height (mm); total
                                      groundwater storage capacity — shallow
                                      buckets are flashier, deep buckets
                                      sustain baseflow longer

From ``divides``:

  areasqkm                          : local catchment area (km2)
  tot_drainage_areasqkm             : total upstream drainage area (km2)

These attributes capture the dominant controls on model performance
differences: terrain (slope, elevation, TWI), soils (porosity, conductivity,
wilting point, matric potential), land use (impervious fraction), runoff
generation (refkdt), snow dynamics (mfsno), groundwater (Coeff, Zmax),
and basin scale (area).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default attribute columns to load from the hydrofabric.
#
# Sourced from two layers: ``divide-attributes`` and ``divides``.
# The loader pulls from both, merges on divide_id, then selects these
# columns.  Modify this list to add or remove attributes without touching
# calling code.
# ---------------------------------------------------------------------------
_DIVIDE_ATTR_COLS: List[str] = [
    "mean.slope",
    "mean.elevation",
    "mean.impervious",
    "mean.refkdt",
    "mean.mfsno",
    "mean.smcmax_soil_layers_stag=1",
    "mean.smcwlt_soil_layers_stag=1",
    "geom_mean.dksat_soil_layers_stag=1",
    "geom_mean.psisat_soil_layers_stag=1",
    "dist_4.twi",
    "mean.Coeff",
    "mean.Zmax",
]

_DIVIDES_COLS: List[str] = [
    "areasqkm",
    "tot_drainage_areasqkm",
]

# The public constant that callers reference — union of both layer sources
DEFAULT_ATTRIBUTE_COLS: List[str] = _DIVIDE_ATTR_COLS + _DIVIDES_COLS

# ID column names that appear in the hydrofabric layers
_DIVIDE_ID_COL    = "divide_id"   # key in divide-attributes AND divides
_FLOWPATH_ID_COL  = "id"          # key in flowpaths / flowpath-attributes


def _extract_numeric_id(series: pd.Series) -> pd.Series:
    """
    Extract the integer numeric part from prefixed hydrofabric IDs.

    Examples:
        "wb-12345"  -> 12345
        "cat-12345" -> 12345
        "12345"     -> 12345

    Returns a Series of integers.
    """
    return (
        series
        .astype(str)
        .str.replace(r"\D+", "", regex=True)
        .astype(float)
        .astype(int)
    )


def load_feature_attributes(
    gpkg_path: Path,
    feature_ids: Optional[List[int]] = None,
    attribute_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Load basin attributes from the hydrofabric and return a DataFrame indexed
    by integer feature_id.

    Pulls attributes from both ``divide-attributes`` (soil/terrain/model
    parameters) and ``divides`` (area, geometry properties), merges on
    ``divide_id``, then joins to flowpath feature_ids via numeric ID
    extraction.

    Parameters
    ----------
    gpkg_path : Path
        Path to the NextGen hydrofabric GeoPackage.
    feature_ids : list of int, optional
        If provided, return attributes only for these feature_ids.  If None,
        return attributes for all flowpaths in the hydrofabric.
    attribute_cols : list of str, optional
        Columns to load from the hydrofabric layers.  Defaults to
        ``DEFAULT_ATTRIBUTE_COLS``.  Unrecognised column names are warned
        about and dropped.

    Returns
    -------
    pd.DataFrame
        Index: feature_id (int)
        Columns: subset of ``attribute_cols`` that were found.
        Rows with all-NaN attributes are kept (caller decides how to handle).
    """
    try:
        import fiona
        import geopandas as gpd
    except ImportError as e:
        raise ImportError(
            "geopandas and fiona are required for attribute loading. "
            f"Original error: {e}"
        )

    gpkg_path = Path(gpkg_path)
    if attribute_cols is None:
        attribute_cols = DEFAULT_ATTRIBUTE_COLS

    # -----------------------------------------------------------------------
    # Discover available layers
    # -----------------------------------------------------------------------
    try:
        available_layers = fiona.listlayers(str(gpkg_path))
    except Exception as e:
        logger.warning(f"Could not open {gpkg_path}: {e}")
        return pd.DataFrame()

    has_divide_attrs = "divide-attributes" in available_layers
    has_divides = "divides" in available_layers

    if not has_divide_attrs and not has_divides:
        logger.warning(
            f"Neither 'divide-attributes' nor 'divides' layer found in "
            f"{gpkg_path.name}. Attribute-conditioned ensemble weights are "
            "unavailable; equal weights will be used as fallback."
        )
        return pd.DataFrame()

    # -----------------------------------------------------------------------
    # Step 1: Build the feature_id -> numeric_id mapping from flowpath layer
    # -----------------------------------------------------------------------
    source_layer = None
    for candidate in ["flowpath-attributes", "flowpaths"]:
        if candidate in available_layers:
            source_layer = candidate
            break

    if source_layer is None:
        logger.warning(
            f"Neither 'flowpath-attributes' nor 'flowpaths' layer found in "
            f"{gpkg_path.name}. Cannot build feature_id mapping."
        )
        return pd.DataFrame()

    logger.info(f"Loading feature_id mapping from '{source_layer}'...")
    try:
        fp_df = gpd.read_file(str(gpkg_path), layer=source_layer)
    except Exception as e:
        logger.warning(f"Could not read {source_layer}: {e}")
        return pd.DataFrame()

    if _FLOWPATH_ID_COL not in fp_df.columns:
        logger.warning(
            f"Column '{_FLOWPATH_ID_COL}' not found in {source_layer}. "
            "Cannot proceed."
        )
        return pd.DataFrame()

    # Determine join strategy
    use_crosswalk = (
        source_layer == "flowpath-attributes"
        and _DIVIDE_ID_COL in fp_df.columns
    )

    if use_crosswalk:
        logger.info(
            f"Using '{_DIVIDE_ID_COL}' crosswalk column in "
            f"{source_layer} for join."
        )
        fp_map = fp_df[[_FLOWPATH_ID_COL, _DIVIDE_ID_COL]].copy()
        fp_map["feature_id"] = _extract_numeric_id(fp_map[_FLOWPATH_ID_COL])
        fp_map = fp_map[["feature_id", _DIVIDE_ID_COL]].drop_duplicates("feature_id")
        join_col = _DIVIDE_ID_COL
    else:
        logger.info(
            f"No '{_DIVIDE_ID_COL}' column in {source_layer}. "
            "Using numeric ID extraction (wb-XXXXX <-> cat-XXXXX) for join."
        )
        fp_map = pd.DataFrame()
        fp_map["feature_id"] = _extract_numeric_id(fp_df[_FLOWPATH_ID_COL])
        fp_map["_numeric_id"] = fp_map["feature_id"]
        fp_map = fp_map.drop_duplicates("feature_id")
        join_col = "_numeric_id"

    if feature_ids is not None:
        fp_map = fp_map[fp_map["feature_id"].isin(feature_ids)]

    logger.debug(f"Feature ID mapping: {len(fp_map)} entries.")

    # -----------------------------------------------------------------------
    # Step 2: Load and merge the two attribute layers
    # -----------------------------------------------------------------------
    attr_frames = []

    # --- divide-attributes (soil, terrain, model parameters) ---
    if has_divide_attrs:
        logger.info("Loading divide-attributes...")
        try:
            div_attr = gpd.read_file(str(gpkg_path), layer="divide-attributes")
            # Keep divide_id + any requested columns that exist in this layer
            da_available = [c for c in attribute_cols if c in div_attr.columns]
            da_missing = [
                c for c in _DIVIDE_ATTR_COLS
                if c in attribute_cols and c not in div_attr.columns
            ]
            if da_missing:
                logger.warning(
                    f"Columns not found in divide-attributes: {da_missing}"
                )
            if da_available:
                div_attr = div_attr[[_DIVIDE_ID_COL] + da_available].copy()
                div_attr = div_attr.drop_duplicates(_DIVIDE_ID_COL)
                attr_frames.append(div_attr)
                logger.info(
                    f"  divide-attributes: {len(div_attr)} rows, "
                    f"{len(da_available)} columns ({da_available})."
                )
        except Exception as e:
            logger.warning(f"Could not read divide-attributes: {e}")

    # --- divides (area, geometry properties) ---
    if has_divides:
        logger.info("Loading divides...")
        try:
            divides = gpd.read_file(str(gpkg_path), layer="divides")
            dv_available = [c for c in attribute_cols if c in divides.columns]
            dv_missing = [
                c for c in _DIVIDES_COLS
                if c in attribute_cols and c not in divides.columns
            ]
            if dv_missing:
                logger.warning(
                    f"Columns not found in divides: {dv_missing}"
                )
            if dv_available:
                divides = divides[[_DIVIDE_ID_COL] + dv_available].copy()
                divides = divides.drop_duplicates(_DIVIDE_ID_COL)
                attr_frames.append(divides)
                logger.info(
                    f"  divides: {len(divides)} rows, "
                    f"{len(dv_available)} columns ({dv_available})."
                )
        except Exception as e:
            logger.warning(f"Could not read divides: {e}")

    if not attr_frames:
        logger.warning(
            "No attribute data loaded from any layer. "
            "Equal weights will be used as fallback."
        )
        return pd.DataFrame()

    # Merge the attribute frames on divide_id
    if len(attr_frames) == 1:
        combined_attrs = attr_frames[0]
    else:
        combined_attrs = attr_frames[0]
        for extra in attr_frames[1:]:
            combined_attrs = combined_attrs.merge(
                extra, on=_DIVIDE_ID_COL, how="outer"
            )

    # Identify which requested columns we actually have
    available_attr_cols = [
        c for c in attribute_cols if c in combined_attrs.columns
    ]
    if not available_attr_cols:
        logger.warning(
            "None of the requested attribute columns were found in any "
            "layer. Equal weights will be used as fallback."
        )
        return pd.DataFrame()

    # -----------------------------------------------------------------------
    # Step 3: Join to feature_ids
    # -----------------------------------------------------------------------
    if use_crosswalk:
        # Crosswalk join: fp_map has divide_id strings
        merged = fp_map.merge(
            combined_attrs[[_DIVIDE_ID_COL] + available_attr_cols],
            on=_DIVIDE_ID_COL,
            how="left",
        )
        merged = merged.set_index("feature_id")
        merged = merged.drop(columns=[_DIVIDE_ID_COL], errors="ignore")

    else:
        # Numeric ID join
        combined_attrs["_numeric_id"] = _extract_numeric_id(
            combined_attrs[_DIVIDE_ID_COL]
        )
        merged = fp_map.merge(
            combined_attrs[["_numeric_id"] + available_attr_cols],
            on="_numeric_id",
            how="left",
        )
        merged = merged.set_index("feature_id")
        merged = merged.drop(columns=["_numeric_id"], errors="ignore")

    # Convert all attribute columns to float
    for col in available_attr_cols:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce")

    n_missing = merged[available_attr_cols].isna().all(axis=1).sum()
    if n_missing > 0:
        logger.warning(
            f"{n_missing} feature_ids have no attribute data after the join "
            "(all attributes are NaN). Equal weights will be used for these."
        )

    logger.info(
        f"Attributes loaded: {len(merged)} feature_ids x "
        f"{len(available_attr_cols)} attributes "
        f"({available_attr_cols})."
    )
    return merged


def normalize_attributes(
    attributes_df: pd.DataFrame,
    center: Optional[pd.Series] = None,
    scale: Optional[pd.Series] = None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Standardize attribute columns to zero mean and unit variance.

    Standardization is important for linear spatial transfer models (ridge
    regression, logistic regression) where features on different scales cause
    numerical instability and make regularization strength non-comparable
    across attributes.

    Parameters
    ----------
    attributes_df : pd.DataFrame
        Raw attribute values, index = feature_id.
    center : pd.Series, optional
        Pre-computed column means (e.g., from training data).  If None,
        computed from ``attributes_df``.
    scale : pd.Series, optional
        Pre-computed column standard deviations.  If None, computed from
        ``attributes_df``.

    Returns
    -------
    normalized : pd.DataFrame
        Standardized attribute values.  NaN rows are preserved as-is.
    center : pd.Series
        Column means used for standardization (save with model artifact).
    scale : pd.Series
        Column standard deviations used for standardization.
    """
    if center is None:
        center = attributes_df.mean()
    if scale is None:
        scale = attributes_df.std().replace(0, 1)  # avoid division by zero

    normalized = (attributes_df - center) / scale
    return normalized, center, scale


def get_gage_attributes(
    attributes_df: pd.DataFrame,
    gage_to_fids: dict,
) -> pd.DataFrame:
    """
    Extract attributes for gaged locations using the primary (outlet) feature_id.

    Given a mapping of gage_id -> list of feature_ids (from the hydrofabric
    crosswalk), return a DataFrame of attributes indexed by gage_id using the
    last (highest-order, outlet) feature_id in each list.

    # NOTE / TODO: Using the outlet feature_id's local catchment attributes
    # is a simplification.  The preferred approach is to use upstream-
    # accumulated attributes for the entire drainage area above the gage.

    Parameters
    ----------
    attributes_df : pd.DataFrame
        Feature-level attributes, index = feature_id (int).
    gage_to_fids : dict
        {gage_id (str): [feature_id (int), ...]} from io.hydrofabric.load_hydrofabric.

    Returns
    -------
    pd.DataFrame
        Index: gage_id (str), columns: same as attributes_df.
        Gages with no attribute data have all-NaN rows.
    """
    rows = {}
    for gage_id, fids in gage_to_fids.items():
        # Use the outlet (last in list from load_hydrofabric ordering)
        outlet_fid = fids[-1]
        if outlet_fid in attributes_df.index:
            rows[gage_id] = attributes_df.loc[outlet_fid]
        else:
            rows[gage_id] = pd.Series(np.nan, index=attributes_df.columns)

    return pd.DataFrame(rows).T