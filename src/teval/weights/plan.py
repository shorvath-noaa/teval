"""
teval.weights.plan

Drive weighting for one domain, in the two phases the run makes available.

Resolution needs four things and the run produces them at two different
moments: the weight file and the nexus crosswalk are known once the hydrofabric
is loaded, while the formulation names and feature ids arrive only when the
formulation files are opened.  A :class:`WeightPlan` carries the first pair
across that gap so the second pair can be joined to it in place.

Both phases run before ``build_stats`` constructs the graph, which is the
property that matters: every weight rule and the coverage policy are decided in
the first second of a run rather than after a long compute.

Public API
----------
WeightPlan
    The file-derived half of a weighted run: config, weight frame, crosswalk.
prepare_weight_plan(stats_config, gdf_hydro, reusing_ensemble=False)
    Phase one, from the hydrofabric.  ``None`` when the run is unweighted.
resolve_domain_weights(plan, combined_ds)
    Phase two, from the opened ensemble: the dense weight array and what to
    record about it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import geopandas as gpd
import pandas as pd
import xarray as xr

from teval.config import StatsConfig, WeightsConfig
from teval.io.hydrofabric import build_nexus_crosswalk
from teval.weights.provenance import AppliedWeighting
from teval.weights.reader import read_weight_file
from teval.weights.resolve import resolve_weights

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WeightPlan:
    """
    The parts of a weighted run that are known before the ensemble is opened.

    Attributes
    ----------
    config:
        The validated ``stats.weights`` block.
    frame:
        Tidy weight frame as ``read_weight_file`` returned it.
    crosswalk:
        Nexus to draining feature ids, as ``build_nexus_crosswalk`` returned it.
        A domain with no hydrofabric never reaches this far -- see
        :func:`prepare_weight_plan` -- so the crosswalk is empty only when a
        hydrofabric was loaded and placed none of its flowpaths, which surfaces
        as zero coverage under the configured ``on_missing`` policy.
    """

    config: WeightsConfig
    frame: pd.DataFrame
    crosswalk: Dict[int, List[int]]


def prepare_weight_plan(
    stats_config: StatsConfig,
    gdf_hydro: gpd.GeoDataFrame,
    reusing_ensemble: bool = False,
) -> Optional[WeightPlan]:
    """
    Read the weight file and build the crosswalk, or return None if unweighted.

    ``None`` means no ``stats.weights`` block was configured, and the caller
    leaves the unweighted code path entirely untouched -- no file is read, no
    crosswalk is built and ``build_stats`` is reached exactly as before.

    *reusing_ensemble* says this domain will return a pre-computed ensemble
    rather than build statistics, as ``workflow.reuses_precomputed_ensemble``
    answers it -- the same call the bypass warning turns on, so the two guard
    rails cannot reach opposite conclusions.  Such a run never consumes the
    crosswalk, so the missing-hydrofabric guard below is not applied to it: the
    accurate complaint there is that weighting is bypassed altogether, which
    ``_process_formulation_files`` makes loudly, and raising instead would
    refuse a configuration the design explicitly permits.

    Raises
    ------
    ValueError
        Weights are configured, statistics will be built, but this domain has
        no hydrofabric.  The crosswalk that turns per-nexus weights into
        per-feature weights is derived from the hydrofabric's flowpaths, so
        without one there is nothing to join the weight file to.  That is a
        configuration mistake rather than a coverage shortfall, so it is
        refused here instead of being handed to ``on_missing`` -- under the
        default 'warn' the run would otherwise complete with an entirely
        unweighted mean, quietly ignoring the file the user supplied.  A
        hydrofabric that *is* present but places no flowpaths does go to the
        coverage policy: there the crosswalk was possible and simply came back
        empty.
    FileNotFoundError
        The configured weight file does not exist.
    """
    weights_config = stats_config.weights
    if weights_config is None:
        return None

    if not reusing_ensemble and (gdf_hydro is None or len(gdf_hydro) == 0):
        raise ValueError(
            f"Ensemble weights are configured (stats.weights.file="
            f"{weights_config.file}), but this domain has no hydrofabric, so "
            f"the nexus-to-feature crosswalk the weights are joined through "
            f"cannot be built. Supply this domain's hydrofabric GeoPackage, or "
            f"remove the stats.weights block to run unweighted. See the "
            f"stats.weights documentation for when teval loads a hydrofabric."
        )

    frame = read_weight_file(weights_config.file)
    crosswalk = build_nexus_crosswalk(gdf_hydro)
    logger.debug(
        f"Read {len(frame)} weight row(s) from {weights_config.file}; the "
        f"hydrofabric crosswalk covers {len(crosswalk)} nexus."
    )
    return WeightPlan(config=weights_config, frame=frame, crosswalk=crosswalk)


def resolve_domain_weights(
    plan: WeightPlan,
    combined_ds: xr.Dataset,
) -> Tuple[xr.DataArray, AppliedWeighting]:
    """
    Resolve the plan against the run's formulations and feature ids.

    Called once per domain, as soon as the combined dataset exists and before
    its statistics graph is built.  Both axes are taken from the dataset's own
    coordinates, unconverted -- the formulation names rather than the raw file
    dict's keys, and the feature ids as they stand -- so the returned array is
    labelled with exactly what ``build_stats`` will match it against, and the
    two cannot drift.

    The coverage report comes back alongside the weights rather than being
    logged and dropped, because the achieved coverage is written into the
    output file as provenance -- see ``teval.weights.provenance``.  It is
    returned already paired with the configuration that produced it, as the
    ``AppliedWeighting`` the provenance step takes, so the caller hands that
    value straight on instead of taking the pair apart and rebuilding it from
    two places.

    Returns
    -------
    (xr.DataArray, AppliedWeighting)
        Dense weights over ``(feature_id, formulation)``, ready to hand to
        ``build_stats``, and what the resolution applied and achieved.

    Raises
    ------
    ValueError
        The dataset lacks a coordinate the resolution needs, or any weight
        rule or the coverage policy rejects the file.
    """
    for coord in ("formulation", "feature_id"):
        if coord not in combined_ds.coords:
            found = ", ".join(str(c) for c in combined_ds.coords) or "(none)"
            raise ValueError(
                f"Ensemble weights are configured, but the combined formulation "
                f"dataset carries no '{coord}' coordinate, so the weights cannot "
                f"be matched to this run. Found coordinate(s): {found}."
            )

    weights, report = resolve_weights(
        plan.frame,
        plan.config.formulation_index_map,
        combined_ds["formulation"].values,
        plan.crosswalk,
        combined_ds["feature_id"].values,
        normalize=plan.config.normalize,
        on_missing=plan.config.on_missing,
    )

    # One summary line per domain, naming the file it came from
    logger.info(
        f"Applying ensemble weights from {plan.config.file}: {report.summary()}. "
        f"Median and the spread band remain unweighted."
    )
    return weights, AppliedWeighting(config=plan.config, report=report)
