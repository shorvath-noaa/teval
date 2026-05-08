"""
teval/ensemble_methods/base.py

Abstract base class and shared utilities for all ensemble combination methods.

All methods in this module follow the same two-phase pattern:

    Phase A — Fit
        Learn combination weights (or model parameters) from paired
        simulation + observation data at gaged locations, conditioned on
        basin attributes.  Produces a trained artifact that can be saved to
        disk.

    Phase B — Predict
        Apply the trained artifact to produce a combined output for every
        feature_id, including ungaged locations where attributes (not
        observations) drive the spatial transfer of weights.

The clean separation between fit and predict maps directly onto the teval
workflow:
    - Multidomain calibration runs → Phase A (fit at ~300–7,000 gaged points)
    - Full CONUS runs              → Phase B (predict at ~800,000 feature_ids)

Seasonal weight estimation
--------------------------
All methods estimate weights per hydrological season rather than over the
full period.  Seasonal weights capture the dominant performance differences
between formulations (e.g., snowmelt formulations that outperform in spring
but not summer; arid-region models that are only skilful during monsoon).

Seasons are defined in the module-level ``SEASONS`` dict:

    SEASONS = {
        "DJF": [12, 1, 2],
        "MAM": [3, 4, 5],
        "JJA": [6, 7, 8],
        "SON": [9, 10, 11],
    }

To extend to monthly resolution, replace each entry with a single month:
    SEASONS = {"Jan": [1], "Feb": [2], ..., "Dec": [12]}

The ``get_season`` utility maps a timestamp to its season label.  All fit
and predict methods iterate over ``SEASONS.keys()`` so the change is
automatic once the dict is updated.

KGE skill threshold
-------------------
``KGE_SKILL_THRESHOLD = 0.0``

Formulations with KGE at or below this value at a given location receive zero
weight before normalization (i.e., they are excluded from the combination at
that location for that season).

KGE = 0 is chosen because it is the boundary below which a model performs no
better than predicting the mean flow at every timestep — in the KGE composite
sense, at least one of (correlation, variability ratio, bias ratio) is
sufficiently poor to drag the composite to zero or below.  A model at
KGE = -0.5 is actively degrading the ensemble at that location.

This is a more conservative exclusion than KGE < -1 (which would allow very
poor models to participate) and is aligned with the operational requirement
that the combined output conserves mass without amplifying errors.

Adjust ``KGE_SKILL_THRESHOLD`` here to change the threshold globally across
all methods.  A future user-configurable parameter could override this.
"""
from __future__ import annotations

import abc
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Seasonal definition
# ---------------------------------------------------------------------------
# To switch to monthly, replace each value with a single-month list:
#   SEASONS = {"Jan": [1], "Feb": [2], ..., "Dec": [12]}
# All downstream code iterates over SEASONS.keys() so no other changes needed.
SEASONS: Dict[str, List[int]] = {
    "DJF": [12, 1, 2],   # Winter
    "MAM": [3, 4, 5],    # Spring
    "JJA": [6, 7, 8],    # Summer
    "SON": [9, 10, 11],  # Fall
}

# ---------------------------------------------------------------------------
# KGE skill threshold for weight zeroing
# ---------------------------------------------------------------------------
# Formulations at or below this threshold at a given location receive zero
# weight before normalisation.  See module docstring for scientific rationale.
# KGE ≤ 0 means the model adds no value relative to a mean-flow predictor.
KGE_SKILL_THRESHOLD: float = 0.0


# ---------------------------------------------------------------------------
# Season utilities
# ---------------------------------------------------------------------------

def get_season(timestamp: pd.Timestamp) -> str:
    """
    Return the season label (from SEASONS) for a given timestamp.

    Raises
    ------
    ValueError
        If the timestamp's month is not covered by any season in SEASONS.
        This should not happen with the default four-season definition but
        is possible with custom SEASONS dicts.
    """
    month = timestamp.month
    for season_name, months in SEASONS.items():
        if month in months:
            return season_name
    raise ValueError(
        f"Month {month} not covered by any season in SEASONS: {SEASONS}. "
        "Check that SEASONS covers all 12 months."
    )


def assign_seasons(index: pd.DatetimeIndex) -> pd.Series:
    """
    Map a DatetimeIndex to season labels.

    Returns a pd.Series with the same index and string season labels as values.
    Vectorized over the month values for performance.
    """
    month_to_season = {}
    for season_name, months in SEASONS.items():
        for m in months:
            month_to_season[m] = season_name
    return pd.Series(index.month, index=index).map(month_to_season)


# ---------------------------------------------------------------------------
# Skill score utilities
# ---------------------------------------------------------------------------

def compute_seasonal_kge(
    sim_df: pd.DataFrame,
    obs_df: pd.DataFrame,
    train_start: str,
    train_end: str,
) -> pd.DataFrame:
    """
    Compute KGE for each (gage_id, formulation, season) triplet over the
    training period.

    Parameters
    ----------
    sim_df : pd.DataFrame
        Simulated flows.
        Index: DatetimeIndex.  Columns: MultiIndex (gage_id, formulation).
    obs_df : pd.DataFrame
        Observed flows.
        Index: DatetimeIndex (same or overlapping with sim_df).
        Columns: gage_id (str).
    train_start : str
        Start of the training period, e.g. '2020-01-01'.
    train_end : str
        End of the training period, e.g. '2020-12-31'.

    Returns
    -------
    pd.DataFrame
        Index: (gage_id, formulation, season) MultiIndex.
        Column: 'kge'.

    Notes
    -----
    KGE = 1 − √[(r − 1)² + (α − 1)² + (β − 1)²]
    where r = Pearson correlation, α = std(sim)/std(obs), β = mean(sim)/mean(obs).

    Gages or seasons with fewer than 10 paired observations return NaN.
    """
    # Slice to training period
    sim_train = sim_df.loc[train_start:train_end]
    obs_train = obs_df.loc[train_start:train_end]

    # Align to common time index — sim and obs may not start/end at the
    # same time (e.g., observations may begin later than simulations).
    # Without this, the seasonal mask computed from one index cannot be
    # applied to the other.
    common_index = sim_train.index.intersection(obs_train.index)
    if len(common_index) == 0:
        logger.warning(
            "No overlapping timesteps between sim and obs in the training "
            f"period ({train_start} → {train_end}). "
            f"Sim range: {sim_train.index.min()} → {sim_train.index.max()}, "
            f"Obs range: {obs_train.index.min()} → {obs_train.index.max()}."
        )
        return pd.DataFrame(columns=["gage_id", "formulation", "season", "kge"])

    sim_train = sim_train.loc[common_index]
    obs_train = obs_train.loc[common_index]

    seasons = assign_seasons(pd.DatetimeIndex(common_index))

    records = []
    gage_ids = obs_train.columns.tolist()
    formulations = sim_train.columns.get_level_values(1).unique().tolist()

    for gage_id in gage_ids:
        if gage_id not in sim_train.columns.get_level_values(0):
            continue
        obs_series = obs_train[gage_id]

        for form in formulations:
            if (gage_id, form) not in sim_train.columns:
                continue
            sim_series = sim_train[(gage_id, form)]

            for season_name in SEASONS:
                mask = seasons == season_name
                o = obs_series[mask].values
                s = sim_series[mask].values

                # Align and drop NaN pairs
                valid = np.isfinite(o) & np.isfinite(s)
                o, s = o[valid].astype(np.float64), s[valid].astype(np.float64)

                if len(o) < 10:
                    kge = np.nan
                else:
                    mean_o, mean_s = np.mean(o), np.mean(s)
                    std_o, std_s   = np.std(o),  np.std(s)
                    if std_o < 1e-6 or mean_o < 1e-6 or std_s < 1e-6:
                        kge = np.nan
                    else:
                        r     = np.corrcoef(o, s)[0, 1]
                        alpha = std_s / std_o
                        beta  = mean_s / mean_o
                        kge   = 1.0 - np.sqrt((r-1)**2 + (alpha-1)**2 + (beta-1)**2)

                records.append({
                    "gage_id":    gage_id,
                    "formulation": form,
                    "season":     season_name,
                    "kge":        kge,
                })

    if not records:
        logger.warning("No KGE records computed — check sim_df/obs_df overlap.")
        return pd.DataFrame(columns=["gage_id", "formulation", "season", "kge"])

    result = (
        pd.DataFrame(records)
        .set_index(["gage_id", "formulation", "season"])
    )
    return result


def apply_skill_threshold(
    kge_series: pd.Series,
    threshold: float = KGE_SKILL_THRESHOLD,
) -> pd.Series:
    """
    Zero out KGE values at or below ``threshold``.

    This is applied before softmax weight computation so that formulations
    with no skill at a given location receive exactly zero weight, preserving
    mass conservation in the convex combination.

    Parameters
    ----------
    kge_series : pd.Series
        KGE values, one per formulation.  May contain NaN (treated as below
        threshold).
    threshold : float
        KGE values ≤ this are zeroed.  Default: KGE_SKILL_THRESHOLD (= 0.0).

    Returns
    -------
    pd.Series
        KGE values with sub-threshold entries replaced by 0.0.
    """
    clipped = kge_series.copy().fillna(-np.inf)
    clipped[clipped <= threshold] = 0.0
    return clipped


def softmax_weights(
    kge_series: pd.Series,
    temperature: float = 1.0,
) -> pd.Series:
    """
    Convert KGE scores to convex combination weights using the softmax function.

    w_i = exp(λ · KGE_i) / Σ exp(λ · KGE_j)

    The softmax ensures weights are always positive and sum to 1 (valid convex
    combination, mass-conserving).  The temperature parameter λ controls
    sharpness:
        λ → 0   : equal weights regardless of skill
        λ = 1.0 : default, moderate sharpening
        λ → ∞   : winner-takes-all (weight on highest-KGE model only)

    Parameters
    ----------
    kge_series : pd.Series
        KGE values per formulation (after threshold zeroing).
    temperature : float
        Sharpness parameter λ.  Default: 1.0.

    Returns
    -------
    pd.Series
        Weights summing to 1.0.  If all KGE values are 0 (all models below
        threshold), returns equal weights as fallback.

    Notes
    -----
    Pre-threshold zeroing means models below KGE_SKILL_THRESHOLD have
    exp(λ · 0) = 1.0 before normalization, giving them a small but non-zero
    weight unless ALL other models also score 0.  To give them exactly zero
    weight, mask after the softmax: weights[kge_series <= threshold] = 0,
    then renormalize.  The ``EnsembleMethod`` base class handles this.
    """
    if kge_series.sum() == 0:
        # All models below threshold — fallback to equal weights
        logger.debug(
            "All formulations at or below KGE threshold for this location "
            "and season. Using equal weights as fallback."
        )
        return pd.Series(1.0 / len(kge_series), index=kge_series.index)

    scores = kge_series * temperature
    # Subtract max for numerical stability before exp
    scores = scores - scores.max()
    exp_scores = np.exp(scores)
    return exp_scores / exp_scores.sum()


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class EnsembleMethod(abc.ABC):
    """
    Abstract base class for all ensemble combination methods.

    Subclasses must implement ``fit`` and ``predict``.  The ``save``/``load``
    methods use joblib by default; subclasses may override for custom formats.

    Class-level attributes (override in subclasses)
    -----------------------------------------------
    name : str
        Human-readable method name used in output file naming and logging.
    conserves_mass : bool
        Whether the method's ``predict`` output is guaranteed to be a
        convex combination of the member outputs (True) or an unconstrained
        prediction (False).  Documented for each method; does not enforce
        the constraint programmatically.
    is_probabilistic : bool
        Whether the method produces a full predictive distribution (True)
        or only a point estimate (False).

    Parameters
    ----------
    train_start : str
        Start of the training period, e.g. '2020-01-01'.  Used in ``fit``.
    train_end : str
        End of the training period, e.g. '2020-12-31'.
    eval_start : str, optional
        Start of the evaluation/test period.  If None, the method is fitted
        but not internally evaluated.
    eval_end : str, optional
        End of the evaluation/test period.
    kge_threshold : float
        Skill threshold below which a formulation receives zero weight at a
        given location.  Default: ``KGE_SKILL_THRESHOLD`` (= 0.0).
    softmax_temperature : float
        Sharpness of the softmax weight function.  Default: 1.0.
    """

    name:            str  = "base"
    conserves_mass:  bool = False
    is_probabilistic: bool = False

    def __init__(
        self,
        train_start: str,
        train_end: str,
        eval_start:  Optional[str] = None,
        eval_end:    Optional[str] = None,
        kge_threshold: float = KGE_SKILL_THRESHOLD,
        softmax_temperature: float = 1.0,
    ) -> None:
        """Initialise the ensemble method with training and evaluation period bounds."""
        self.train_start = train_start
        self.train_end   = train_end
        self.eval_start  = eval_start
        self.eval_end    = eval_end
        self.kge_threshold       = kge_threshold
        self.softmax_temperature = softmax_temperature
        self._is_fitted = False
        self._logger = logging.getLogger(
            f"teval.ensemble_methods.{self.__class__.__name__}"
        )

    @property
    def is_fitted(self) -> bool:
        """True after ``fit`` has been called successfully."""
        return self._is_fitted

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def fit(
        self,
        sim_df: pd.DataFrame,
        obs_df: pd.DataFrame,
        attributes_df: pd.DataFrame,
    ) -> "EnsembleMethod":
        """
        Learn combination weights from paired simulation and observation data.

        Parameters
        ----------
        sim_df : pd.DataFrame
            Simulated flows at GAGED locations.
            Index: DatetimeIndex.
            Columns: MultiIndex(level 0 = gage_id, level 1 = formulation).
            The MultiIndex allows multiple gages and formulations to be passed
            simultaneously, as produced by the teval multidomain workflow.

        obs_df : pd.DataFrame
            Observed flows at gaged locations.
            Index: DatetimeIndex (overlapping with sim_df).
            Columns: gage_id (str, zero-padded 8-digit USGS site numbers).

        attributes_df : pd.DataFrame
            Basin attributes at gaged locations.
            Index: gage_id (str, matching obs_df columns).
            Columns: attribute names (from attributes.DEFAULT_ATTRIBUTE_COLS).
            Used to learn the spatial transfer from gaged to ungaged locations.
            Rows with all-NaN attributes are handled with the fallback path.

        Returns
        -------
        self
            The fitted method instance (for method chaining).
        """
        ...

    @abc.abstractmethod
    def predict(
        self,
        sim_df: pd.DataFrame,
        attributes_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Apply the trained weights to produce a combined flow estimate.

        Parameters
        ----------
        sim_df : pd.DataFrame
            Simulated flows for ALL feature_ids (including ungaged).
            Index: DatetimeIndex.
            Columns: MultiIndex(level 0 = feature_id, level 1 = formulation).

        attributes_df : pd.DataFrame
            Basin attributes for ALL feature_ids.
            Index: feature_id (int).
            Columns: attribute names.
            feature_ids with all-NaN attributes fall back to equal weights.

        Returns
        -------
        pd.DataFrame
            Combined flow estimates.
            Index: DatetimeIndex (same as sim_df).
            Columns: feature_id (int).
            Shape matches a single formulation's output — one value per
            feature_id per timestep — so it slots directly into the existing
            ensemble NC writing infrastructure.

        Raises
        ------
        RuntimeError
            If called before ``fit``.
        """
        ...

    # ------------------------------------------------------------------
    # Serialization (default: joblib)
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        """
        Save the fitted method to disk using joblib.

        The saved artifact can be loaded with ``EnsembleMethod.load`` and
        applied to new simulation data without re-fitting.

        Parameters
        ----------
        path : Path
            Output file path.  Convention: ``{method_name}_artifact.joblib``.
        """
        try:
            import joblib
        except ImportError:
            raise ImportError("joblib is required for save/load. pip install joblib")

        if not self._is_fitted:
            raise RuntimeError("Cannot save: method has not been fitted yet.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        self._logger.info(f"Artifact saved → {path}")

    @classmethod
    def load(cls, path: Path) -> "EnsembleMethod":
        """
        Load a previously saved method artifact from disk.

        Parameters
        ----------
        path : Path
            Path to a ``.joblib`` file saved with ``save``.

        Returns
        -------
        EnsembleMethod
            The fitted method instance, ready for ``predict``.
        """
        try:
            import joblib
        except ImportError:
            raise ImportError("joblib is required for save/load. pip install joblib")
        instance = joblib.load(path)
        if not isinstance(instance, EnsembleMethod):
            raise TypeError(
                f"Loaded object is {type(instance)}, not an EnsembleMethod subclass."
            )
        logger.info(f"Artifact loaded ← {path}  (method: {instance.name})")
        return instance

    # ------------------------------------------------------------------
    # Shared helper: per-season weight computation
    # ------------------------------------------------------------------

    def _compute_raw_weights(
        self,
        kge_by_season: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Convert seasonal KGE scores to softmax weights per (gage_id, season).

        This is a shared utility for all methods that use performance-based
        weighting.  Applies the skill threshold and softmax transformation.

        Parameters
        ----------
        kge_by_season : pd.DataFrame
            MultiIndex (gage_id, formulation, season) → kge column.
            As returned by ``compute_seasonal_kge``.

        Returns
        -------
        pd.DataFrame
            Index: (gage_id, season) MultiIndex.
            Columns: formulation names.
            Values: softmax weights summing to 1.0 per row.
        """
        formulations = kge_by_season.index.get_level_values("formulation").unique()
        gage_ids     = kge_by_season.index.get_level_values("gage_id").unique()
        seasons      = list(SEASONS.keys())

        records = []
        for gage_id in gage_ids:
            for season in seasons:
                try:
                    kge_row = kge_by_season.loc[(gage_id, slice(None), season), "kge"]
                    kge_row.index = kge_row.index.get_level_values("formulation")
                except KeyError:
                    # No data for this gage/season combination
                    kge_row = pd.Series(np.nan, index=formulations)

                # Apply skill threshold: zero out below-threshold models
                clipped = apply_skill_threshold(kge_row, self.kge_threshold)

                # Compute softmax weights
                weights = softmax_weights(clipped, self.softmax_temperature)

                # Zero out weights for models that were below threshold
                # (softmax of 0 is non-zero; we want exactly 0 for bad models)
                below_threshold = kge_row.fillna(-np.inf) <= self.kge_threshold
                weights[below_threshold] = 0.0
                # Renormalize after zeroing
                total = weights.sum()
                if total > 0:
                    weights = weights / total
                else:
                    weights = pd.Series(1.0 / len(formulations), index=formulations)

                row = {"gage_id": gage_id, "season": season}
                row.update(weights.to_dict())
                records.append(row)

        result = pd.DataFrame(records).set_index(["gage_id", "season"])
        return result

    # ------------------------------------------------------------------
    # Shared helper: predict-time season lookup
    # ------------------------------------------------------------------

    def _get_weights_for_timestep(
        self,
        timestamp: pd.Timestamp,
        feature_id: int,
        weight_table: pd.DataFrame,
        fallback_n_formulations: int,
    ) -> pd.Series:
        """
        Look up the weight vector for a given timestep and feature_id.

        Matches by season (not exact timestamp).  Falls back to equal weights
        if the feature_id has no entry in the weight table.

        Parameters
        ----------
        timestamp : pd.Timestamp
        feature_id : int
        weight_table : pd.DataFrame
            Index: (feature_id, season) MultiIndex.
            Columns: formulation names.
        fallback_n_formulations : int
            Number of formulations for constructing equal-weight fallback.

        Returns
        -------
        pd.Series
            Weights, one per formulation.
        """
        season = get_season(timestamp)
        try:
            return weight_table.loc[(feature_id, season)]
        except KeyError:
            formulations = weight_table.columns
            return pd.Series(
                1.0 / fallback_n_formulations,
                index=formulations,
            )

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """Return a concise string representation showing name, mass-conservation, and fit status."""
        status = "fitted" if self._is_fitted else "unfitted"
        return (
            f"{self.__class__.__name__}("
            f"name='{self.name}', "
            f"conserves_mass={self.conserves_mass}, "
            f"train={self.train_start}→{self.train_end}, "
            f"status={status})"
        )