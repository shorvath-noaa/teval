"""
teval/ensemble_methods/weighted_mean.py

Performance-weighted ensemble mean — Phase 1 ensemble combination method.

This is the simplest departure from equal weighting and produces a convex
combination of member flows, conserving mass and making it suitable for the
operational NWM product.

Algorithm overview
------------------

**Fit** (at gaged locations, typically multidomain calibration runs):

    1. Compute seasonal KGE for each (gage, formulation, season) triplet over
       the training period.
    2. Apply skill threshold + softmax to get raw performance weights at each
       gaged location.  This yields a weight table indexed by (gage_id, season)
       with one column per formulation.
    3. Fit a ridge regression per (formulation, season) to learn the mapping
       from basin attributes to weights:
           weight_i(season) = X_attr @ β_i(season) + β_0_i(season)
       where X_attr is the standardized attribute matrix at gaged locations.
       Ridge regression (L2 penalty) prevents overfitting when the number of
       gages is comparable to the number of attributes, and ensures stable
       weight predictions in attribute space regions with sparse gages.
    4. Store the trained regression coefficients, normalization parameters
       (center/scale from training attributes), and the raw gage-level weight
       table as the fitted artifact.

**Predict** (at all feature_ids, typically full CONUS):

    1. Standardize attributes at all feature_ids using the training-set
       center/scale (critical: do NOT refit normalization on CONUS data).
    2. For each feature_id, predict the weight vector from its attributes
       via the ridge regression.  Clip negative predictions to zero and
       renormalize to ensure convexity (mass conservation).
    3. For feature_ids with missing attributes (all-NaN), fall back to
       equal weights.
    4. Apply the seasonal weight vectors to the formulation time series
       to produce the combined output.

Spatial transfer
----------------
The ridge regression is the spatial transfer mechanism.  At gaged locations
we have ground-truth weights derived from observed performance.  Basin
attributes (aridity, snow fraction, baseflow index, etc.) encode the physical
characteristics that drive performance differences.  The regression learns
which attributes predict where each formulation does well.  At ungaged
locations, we use the same attributes to predict the weight vector without
needing observations.

This is a linear spatial transfer — the simplest reasonable model.  Phase 4
(RF/GBT meta-learners) will extend this to nonlinear attribute–weight
relationships.

Fallback behavior
-----------------
Equal weights are used when:
  - A feature_id has all-NaN attributes (missing hydrofabric data)
  - The ridge regression produces a fully-zero weight vector after clipping
  - All formulations are below the KGE skill threshold at a location
  - Attribute data is entirely unavailable (empty attributes_df)

In these cases the method degrades gracefully to the existing equal-weight
mean, which is the current teval default.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from teval.ensemble_methods.base import (
    SEASONS,
    EnsembleMethod,
    compute_seasonal_kge,
)
from teval.experimental.attributes import normalize_attributes

logger = logging.getLogger(__name__)


class PerformanceWeightedMean(EnsembleMethod):
    """
    Seasonal, attribute-conditioned performance-weighted ensemble mean.

    Mass-conserving: the output at every feature_id and timestep is a convex
    combination (non-negative weights summing to 1) of the member formulation
    outputs.

    Parameters
    ----------
    train_start, train_end : str
        Training period for seasonal KGE computation.
    eval_start, eval_end : str, optional
        Evaluation period (not used during fit, but stored for later
        skill assessment by the evaluate harness).
    kge_threshold : float
        Formulations with KGE at or below this threshold receive zero
        weight at that location/season.  Default: 0.0.
    softmax_temperature : float
        Sharpness of the softmax weight function (λ).
        λ → 0: equal weights; λ → ∞: winner-takes-all.  Default: 1.0.
    ridge_alpha : float
        L2 regularization strength for the attribute → weight regression.
        Higher values = more regularization = weights closer to the
        gage-average.  Default: 1.0.  Reasonable range: 0.01–100.
    min_gages_for_regression : int
        Minimum number of gages with valid weights required to fit the
        ridge regression.  Below this threshold, the method falls back
        to the gage-average weight vector (no spatial transfer).
        Default: 10.

    Attributes (after fit)
    ----------------------
    gage_weights_ : pd.DataFrame
        Raw performance weights at gaged locations.
        Index: (gage_id, season).  Columns: formulation names.
    regression_coefs_ : dict
        {season: {formulation: (coef_array, intercept)}}
        Ridge regression coefficients for spatial transfer.
    attr_center_ : pd.Series
        Training-set attribute means (for standardization at predict time).
    attr_scale_ : pd.Series
        Training-set attribute standard deviations.
    formulations_ : list of str
        Ordered list of formulation names.
    fallback_weights_ : dict
        {season: pd.Series} — gage-average weights per season, used as
        fallback for feature_ids with missing attributes.
    """

    name = "performance_weighted_mean"
    conserves_mass = True
    is_probabilistic = False

    def __init__(
        self,
        train_start: str,
        train_end: str,
        eval_start: Optional[str] = None,
        eval_end: Optional[str] = None,
        kge_threshold: float = 0.0,
        softmax_temperature: float = 1.0,
        ridge_alpha: float = 1.0,
        min_gages_for_regression: int = 10,
    ) -> None:
        """Initialise the PerformanceWeightedMean with training period and hyperparameters."""
        super().__init__(
            train_start=train_start,
            train_end=train_end,
            eval_start=eval_start,
            eval_end=eval_end,
            kge_threshold=kge_threshold,
            softmax_temperature=softmax_temperature,
        )
        self.ridge_alpha = ridge_alpha
        self.min_gages_for_regression = min_gages_for_regression

        # Fitted state (populated by fit())
        self.gage_weights_: Optional[pd.DataFrame] = None
        self.regression_coefs_: Optional[Dict] = None
        self.attr_center_: Optional[pd.Series] = None
        self.attr_scale_: Optional[pd.Series] = None
        self.formulations_: Optional[List[str]] = None
        self.fallback_weights_: Optional[Dict] = None
        self.attribute_cols_: Optional[List[str]] = None

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        sim_df: pd.DataFrame,
        obs_df: pd.DataFrame,
        attributes_df: pd.DataFrame,
    ) -> "PerformanceWeightedMean":
        """
        Learn performance weights and the attribute → weight regression.

        Steps
        -----
        1. compute_seasonal_kge at gaged locations over training period
        2. _compute_raw_weights → softmax weights per (gage_id, season)
        3. Normalize attributes at gaged locations
        4. Fit ridge regression per (season, formulation)
        5. Compute fallback (gage-average) weights per season

        See class docstring for full algorithm description.
        """
        self._logger.info(
            f"Fitting {self.name}: "
            f"train={self.train_start}→{self.train_end}, "
            f"λ={self.softmax_temperature}, α_ridge={self.ridge_alpha}"
        )

        # --- Step 1: Seasonal KGE -------------------------------------------
        self._logger.info("Computing seasonal KGE at gaged locations...")
        kge_df = compute_seasonal_kge(
            sim_df, obs_df, self.train_start, self.train_end
        )

        if kge_df.empty:
            raise ValueError(
                "No KGE scores could be computed. Check that sim_df and "
                "obs_df overlap in time and share common gage_ids."
            )

        self.formulations_ = sorted(
            kge_df.index.get_level_values("formulation").unique().tolist()
        )
        n_forms = len(self.formulations_)
        n_gages = len(kge_df.index.get_level_values("gage_id").unique())
        self._logger.info(
            f"KGE computed: {n_gages} gages × {n_forms} formulations × "
            f"{len(SEASONS)} seasons."
        )

        # --- Step 2: Raw performance weights ---------------------------------
        self._logger.info("Computing softmax weights from KGE scores...")
        self.gage_weights_ = self._compute_raw_weights(kge_df)

        # Log weight diagnostics
        for season in SEASONS:
            season_weights = self.gage_weights_.xs(season, level="season")
            self._logger.info(
                f"  {season}: mean weights = "
                f"{season_weights[self.formulations_].mean().to_dict()}"
            )

        # --- Step 3: Normalize attributes ------------------------------------
        has_attributes = (
            attributes_df is not None
            and not attributes_df.empty
        )

        if has_attributes:
            # Align attributes to gages that have weights
            gage_ids_with_weights = (
                self.gage_weights_.index.get_level_values("gage_id").unique()
            )
            # attributes_df may be indexed by gage_id (str) or feature_id (int)
            # depending on whether caller used get_gage_attributes or not
            common_gages = gage_ids_with_weights.intersection(
                attributes_df.index
            )

            if len(common_gages) < self.min_gages_for_regression:
                self._logger.warning(
                    f"Only {len(common_gages)} gages have both weights and "
                    f"attributes (need {self.min_gages_for_regression}). "
                    "Spatial transfer disabled; gage-average weights will be "
                    "used everywhere."
                )
                has_attributes = False
            else:
                attr_gages = attributes_df.loc[common_gages].copy()
                # Drop gages where ALL attributes are NaN
                attr_gages = attr_gages.dropna(how="all")

                if len(attr_gages) < self.min_gages_for_regression:
                    self._logger.warning(
                        f"Only {len(attr_gages)} gages have non-null "
                        f"attributes after dropping all-NaN rows. "
                        "Spatial transfer disabled."
                    )
                    has_attributes = False

        if has_attributes:
            self._logger.info(
                f"Normalizing attributes for {len(attr_gages)} gages..."
            )
            attr_norm, self.attr_center_, self.attr_scale_ = (
                normalize_attributes(attr_gages)
            )
            self.attribute_cols_ = list(attr_norm.columns)

            # Fill remaining per-column NaNs with 0 (= population mean after
            # standardization) so the regression can use partial attribute data
            attr_norm = attr_norm.fillna(0.0)

            # --- Step 4: Ridge regression per (season, formulation) ----------
            self._logger.info(
                f"Fitting ridge regression (α={self.ridge_alpha}) for "
                f"spatial transfer: {len(self.attribute_cols_)} attributes × "
                f"{n_forms} formulations × {len(SEASONS)} seasons..."
            )
            self.regression_coefs_ = self._fit_ridge(attr_norm)
        else:
            self._logger.info(
                "No usable attributes — spatial transfer disabled. "
                "Predict will use gage-average weights at all locations."
            )
            self.regression_coefs_ = None
            self.attr_center_ = None
            self.attr_scale_ = None
            self.attribute_cols_ = None

        # --- Step 5: Fallback weights (gage-average per season) --------------
        self.fallback_weights_ = {}
        for season in SEASONS:
            season_weights = self.gage_weights_.xs(season, level="season")
            self.fallback_weights_[season] = (
                season_weights[self.formulations_].mean()
            )

        self._is_fitted = True
        self._logger.info(f"Fit complete. Artifact ready for save/predict.")
        return self

    def _fit_ridge(
        self,
        attr_norm: pd.DataFrame,
    ) -> Dict:
        """
        Fit ridge regression: weight ~ attributes, per (season, formulation).

        Uses the closed-form solution rather than sklearn to avoid the
        dependency:
            β = (X^T X + α I)^{-1} X^T y

        Parameters
        ----------
        attr_norm : pd.DataFrame
            Standardized attributes at gaged locations.
            Index: gage_id.  Columns: attribute names.

        Returns
        -------
        dict
            {season: {formulation: (coef, intercept)}}
            where coef is a 1-D array of length n_attributes and
            intercept is a scalar.
        """
        coefs = {}
        n_attrs = len(self.attribute_cols_)

        for season in SEASONS:
            coefs[season] = {}
            season_weights = self.gage_weights_.xs(season, level="season")

            # Align: only gages present in both weights and attributes
            common = season_weights.index.intersection(attr_norm.index)
            if len(common) < self.min_gages_for_regression:
                self._logger.debug(
                    f"  {season}: only {len(common)} gages — using mean "
                    f"weights (no regression)."
                )
                for form in self.formulations_:
                    coefs[season][form] = (
                        np.zeros(n_attrs),
                        season_weights[form].mean(),
                    )
                continue

            X = attr_norm.loc[common].values  # (n_gages, n_attrs)

            # Add intercept column
            ones = np.ones((X.shape[0], 1))
            X_aug = np.hstack([X, ones])  # (n_gages, n_attrs + 1)

            # Ridge penalty on coefficients only, not intercept
            penalty = np.eye(n_attrs + 1) * self.ridge_alpha
            penalty[-1, -1] = 0.0  # no penalty on intercept

            # (X^T X + α I)^{-1} X^T
            XtX = X_aug.T @ X_aug
            try:
                solve_matrix = np.linalg.solve(XtX + penalty, X_aug.T)
            except np.linalg.LinAlgError:
                self._logger.warning(
                    f"  {season}: singular matrix — falling back to mean "
                    f"weights for all formulations."
                )
                for form in self.formulations_:
                    coefs[season][form] = (
                        np.zeros(n_attrs),
                        season_weights[form].mean(),
                    )
                continue

            for form in self.formulations_:
                y = season_weights.loc[common, form].values  # (n_gages,)
                beta = solve_matrix @ y  # (n_attrs + 1,)
                coefs[season][form] = (beta[:-1], beta[-1])

            self._logger.debug(
                f"  {season}: ridge fit on {len(common)} gages, "
                f"R² range: "
                f"{self._quick_r2(X_aug, season_weights, common, coefs[season])}"
            )

        return coefs

    def _quick_r2(
        self,
        X_aug: np.ndarray,
        season_weights: pd.DataFrame,
        common_gages,
        season_coefs: dict,
    ) -> str:
        """Quick training R² diagnostic for logging."""
        r2s = []
        for form in self.formulations_:
            y = season_weights.loc[common_gages, form].values
            coef, intercept = season_coefs[form]
            y_pred = X_aug[:, :-1] @ coef + intercept
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
            r2s.append(r2)
        return f"[{min(r2s):.3f}, {max(r2s):.3f}]"

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(
        self,
        sim_df: pd.DataFrame,
        attributes_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Apply trained weights to produce the combined flow estimate.

        For each feature_id and timestep:
            combined_flow = Σ_i  w_i(season, feature_id) · sim_i

        where w_i is predicted from basin attributes via the ridge
        regression (or fallback weights if attributes are unavailable).

        Parameters
        ----------
        sim_df : pd.DataFrame
            Simulated flows for ALL feature_ids.
            Index: DatetimeIndex.
            Columns: MultiIndex (feature_id, formulation).
        attributes_df : pd.DataFrame
            Basin attributes for ALL feature_ids.
            Index: feature_id (int).
            Columns: attribute names.

        Returns
        -------
        pd.DataFrame
            Combined flow.  Index: DatetimeIndex.  Columns: feature_id.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "PerformanceWeightedMean has not been fitted. "
                "Call fit() before predict()."
            )

        feature_ids = sim_df.columns.get_level_values(0).unique()
        formulations = sim_df.columns.get_level_values(1).unique().tolist()
        n_forms = len(formulations)

        # Validate formulation alignment
        missing = set(self.formulations_) - set(formulations)
        if missing:
            raise ValueError(
                f"Formulations in sim_df do not match training: "
                f"missing {missing}. "
                f"Expected: {self.formulations_}, got: {formulations}"
            )

        # --- Predict weight table for all feature_ids -----------------------
        self._logger.info(
            f"Predicting weights for {len(feature_ids)} feature_ids..."
        )
        weight_table = self._predict_weights(feature_ids, attributes_df)

        # --- Apply weights per season ---------------------------------------
        self._logger.info(
            f"Applying seasonal weights to {len(sim_df)} timesteps × "
            f"{len(feature_ids)} feature_ids..."
        )
        result = self._apply_weights(sim_df, weight_table, feature_ids)

        self._logger.info("Predict complete.")
        return result

    def _predict_weights(
        self,
        feature_ids,
        attributes_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Predict the (feature_id, season) → weight vector table.

        Uses ridge regression if fitted, otherwise fallback weights.

        Returns
        -------
        pd.DataFrame
            Index: (feature_id, season) MultiIndex.
            Columns: formulation names.
        """
        n_forms = len(self.formulations_)
        records = []

        # Prepare normalized attributes if regression is available
        attr_norm = None
        if self.regression_coefs_ is not None and attributes_df is not None:
            if not attributes_df.empty and self.attribute_cols_ is not None:
                # Use only the columns the regression was trained on
                available = [
                    c for c in self.attribute_cols_
                    if c in attributes_df.columns
                ]
                if available:
                    attr_subset = attributes_df[available].copy()
                    attr_norm, _, _ = normalize_attributes(
                        attr_subset,
                        center=self.attr_center_[available],
                        scale=self.attr_scale_[available],
                    )
                    # Fill NaNs with 0 (= population mean)
                    attr_norm = attr_norm.fillna(0.0)

        use_regression = (
            self.regression_coefs_ is not None and attr_norm is not None
        )

        # Track statistics for logging
        n_regression = 0
        n_fallback = 0

        for fid in feature_ids:
            for season in SEASONS:
                if use_regression and fid in attr_norm.index:
                    # Check if this feature_id had all-NaN attributes
                    # BEFORE normalization (they'd now be 0.0, but if
                    # the original was all-NaN we should fallback)
                    if (
                        attributes_df is not None
                        and fid in attributes_df.index
                        and attributes_df.loc[fid].isna().all()
                    ):
                        weights = self.fallback_weights_[season]
                        n_fallback += 1
                    else:
                        weights = self._predict_single(
                            attr_norm.loc[fid].values, season
                        )
                        n_regression += 1
                else:
                    weights = self.fallback_weights_[season]
                    n_fallback += 1

                row = {"feature_id": fid, "season": season}
                for i, form in enumerate(self.formulations_):
                    row[form] = weights[i] if isinstance(weights, np.ndarray) else weights[form]
                records.append(row)

        n_total = n_regression + n_fallback
        if n_total > 0:
            self._logger.info(
                f"  Weights predicted: {n_regression}/{n_total} via "
                f"regression, {n_fallback}/{n_total} via fallback."
            )

        result = pd.DataFrame(records).set_index(["feature_id", "season"])
        return result

    def _predict_single(
        self,
        x_norm: np.ndarray,
        season: str,
    ) -> np.ndarray:
        """
        Predict the weight vector for a single feature_id and season.

        Applies the ridge regression, clips negatives to zero, and
        renormalizes to ensure convexity.

        Parameters
        ----------
        x_norm : np.ndarray
            Standardized attribute vector, shape (n_attributes,).
        season : str
            Season label.

        Returns
        -------
        np.ndarray
            Weight vector, shape (n_formulations,).  Non-negative, sums to 1.
        """
        weights = np.zeros(len(self.formulations_))

        for i, form in enumerate(self.formulations_):
            coef, intercept = self.regression_coefs_[season][form]
            weights[i] = x_norm @ coef + intercept

        # Clip negatives to zero (regression can produce negative predictions)
        weights = np.clip(weights, 0.0, None)

        # Renormalize to ensure convexity (mass conservation)
        total = weights.sum()
        if total > 1e-12:
            weights = weights / total
        else:
            # All predictions are zero/negative → fallback to equal weights
            weights = np.full(len(self.formulations_), 1.0 / len(self.formulations_))

        return weights

    def _apply_weights(
        self,
        sim_df: pd.DataFrame,
        weight_table: pd.DataFrame,
        feature_ids,
    ) -> pd.DataFrame:
        """
        Apply the seasonal weight table to the simulation data.

        Vectorized per-season: for each season, extract all timesteps
        belonging to that season, multiply each formulation's values by
        its weight, and sum.  This avoids a per-timestep Python loop.

        Parameters
        ----------
        sim_df : pd.DataFrame
            Simulated flows.
            Index: DatetimeIndex.
            Columns: MultiIndex (feature_id, formulation).
        weight_table : pd.DataFrame
            Index: (feature_id, season).
            Columns: formulation names.
        feature_ids : Index
            feature_ids to process.

        Returns
        -------
        pd.DataFrame
            Combined flow.  Index: DatetimeIndex.  Columns: feature_id.
        """
        from teval.ensemble_methods.base import assign_seasons

        timestamps = pd.DatetimeIndex(sim_df.index)
        seasons_series = assign_seasons(timestamps)

        # Pre-allocate output array
        result = np.zeros((len(timestamps), len(feature_ids)), dtype=np.float64)
        fid_to_col = {fid: i for i, fid in enumerate(feature_ids)}

        for season in SEASONS:
            season_mask = (seasons_series == season).values
            if not season_mask.any():
                continue

            n_timesteps = season_mask.sum()

            for fid in feature_ids:
                col_idx = fid_to_col[fid]

                try:
                    w_row = weight_table.loc[(fid, season)]
                except KeyError:
                    # No weight entry — equal weights
                    w_row = pd.Series(
                        1.0 / len(self.formulations_),
                        index=self.formulations_,
                    )

                # Extract simulation values for this feature_id, all
                # formulations, for this season's timesteps
                # sim_df columns are MultiIndex (feature_id, formulation)
                combined = np.zeros(n_timesteps, dtype=np.float64)
                for form in self.formulations_:
                    w = w_row[form] if form in w_row.index else 0.0
                    if abs(w) < 1e-12:
                        continue
                    sim_vals = sim_df.loc[season_mask, (fid, form)].values
                    combined += w * sim_vals.astype(np.float64)

                result[season_mask, col_idx] = combined

        return pd.DataFrame(
            result,
            index=timestamps,
            columns=feature_ids,
        )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_weight_summary(self) -> pd.DataFrame:
        """
        Return a summary of gage-level weights for inspection.

        Returns
        -------
        pd.DataFrame
            One row per (gage_id, season), columns = formulation weights.
            Only available after fit().
        """
        if not self._is_fitted:
            raise RuntimeError("Not fitted yet.")
        return self.gage_weights_.copy()

    def get_regression_diagnostics(self) -> pd.DataFrame:
        """
        Return regression coefficient magnitudes for interpretability.

        Returns
        -------
        pd.DataFrame
            Index: attribute names.
            Columns: MultiIndex (season, formulation).
            Values: regression coefficients (not intercepts).
        """
        if not self._is_fitted or self.regression_coefs_ is None:
            raise RuntimeError(
                "Not fitted or no regression was performed "
                "(attributes may have been unavailable)."
            )

        records = []
        for season in SEASONS:
            for form in self.formulations_:
                coef, intercept = self.regression_coefs_[season][form]
                row = {
                    "season": season,
                    "formulation": form,
                    "intercept": intercept,
                }
                for i, attr_name in enumerate(self.attribute_cols_):
                    row[attr_name] = coef[i]
                records.append(row)

        return pd.DataFrame(records).set_index(["season", "formulation"])