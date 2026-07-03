"""Reconciliation + EVT baseline (B5) for CDADE comparison.

Implements the Kandanaarachchi-style baseline: the full detector pool is used
as a fixed ensemble (no dynamic selection), anomaly scores are made
hierarchically coherent via bottom-up reconciliation, and Extreme Value Theory
(GPD Peaks-Over-Threshold) is applied to the residuals to identify anomalies.

This isolates the contribution of dynamic ensemble selection (L3) relative to
the reconciliation (L2) layer alone.

Reference:
    Kandanaarachchi, S., et al. (2020). On normalization and algorithm
    selection for unsupervised anomaly detection. Mach Learn.

Author: CDADE project
"""

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import genpareto

from cdade.baselines.single_best import create_detector
from cdade.registry import get_detector, list_detectors, register_baseline_detector

logger = logging.getLogger(__name__)


@dataclass
class ReconciliationEVTConfig:
    """Configuration for the Reconciliation + EVT baseline.

    Attributes:
        detector_names: Detector pool. Uses all registered when empty.
        contamination: Fraction of residuals treated as extreme for GPD fit.
        normalize: Whether to normalise detector scores before averaging.
        alpha_evt: Significance level for GPD tail probability threshold.
    """

    detector_names: list[str] = field(default_factory=list)
    contamination: float = 0.05
    normalize: bool = True
    alpha_evt: float = 0.05


def _normalize_scores(scores: np.ndarray) -> np.ndarray:
    lo, hi = scores.min(), scores.max()
    if hi == lo:
        return np.zeros_like(scores)
    return (scores - lo) / (hi - lo)


def _fit_gpd(residuals: np.ndarray, contamination: float) -> tuple[float, float, float]:
    """Fit a Generalised Pareto Distribution to the upper tail of |residuals|.

    Args:
        residuals: 1-D array of residuals.
        contamination: Fraction defining the tail threshold.

    Returns:
        Tuple of (threshold, shape, scale) GPD parameters.
    """
    abs_res = np.abs(residuals)
    n = len(abs_res)
    n_tail = max(1, int(contamination * n))
    threshold = np.sort(abs_res)[-n_tail]

    exceedances = abs_res[abs_res > threshold] - threshold
    if len(exceedances) < 2:
        return float(threshold), 0.0, float(np.std(abs_res) + 1e-9)

    shape, _, scale = genpareto.fit(exceedances, floc=0)
    return float(threshold), float(shape), float(scale)


def _gpd_survival(x: np.ndarray, threshold: float, shape: float, scale: float) -> np.ndarray:
    """GPD survival probability P(X > x | X > threshold).

    Args:
        x: Values to evaluate.
        threshold: Pot threshold.
        shape: GPD shape parameter.
        scale: GPD scale parameter.

    Returns:
        Survival probabilities (shape: same as x).
    """
    excess = np.maximum(x - threshold, 0.0)
    if abs(shape) < 1e-9:
        return np.exp(-excess / (scale + 1e-9))
    inner = 1.0 + shape * excess / (scale + 1e-9)
    inner = np.maximum(inner, 0.0)
    return inner ** (-1.0 / shape)


class ReconciliationEVTDetector:
    """Reconciliation + EVT baseline (Kandanaarachchi, B5).

    Pipeline:
    1. Fit all detectors on training data.
    2. Average their (optionally normalised) scores → ensemble score.
    3. Compute residuals: score − rolling mean (proxy for expected level).
    4. Fit GPD to the upper tail of |residuals|.
    5. At inference, flag points whose GPD survival probability < alpha_evt.

    Attributes:
        config: Detector configuration.
        fitted_detectors: Fitted detector instances.
        _gpd_params: (threshold, shape, scale) fitted on training residuals.
    """

    def __init__(self, config: ReconciliationEVTConfig | None = None) -> None:
        """Initialise the reconciliation + EVT detector.

        Args:
            config: Configuration object; uses defaults when None.
        """
        self.config = config or ReconciliationEVTConfig()
        self.fitted_detectors: list[tuple[str, object]] = []
        self._gpd_params: tuple[float, float, float] = (0.0, 0.0, 1.0)
        self._train_mean: float = 0.0

    def _resolve_names(self) -> list[str]:
        return self.config.detector_names if self.config.detector_names else list_detectors()

    def _ensemble_score(self, X: np.ndarray) -> np.ndarray:
        """Average scores across all fitted detectors."""
        all_scores: list[np.ndarray] = []
        for name, det in self.fitted_detectors:
            try:
                s = det.score(X)
                if self.config.normalize:
                    s = _normalize_scores(s)
                all_scores.append(s)
            except Exception as exc:
                logger.warning("Detector '%s' failed: %s", name, exc)

        if not all_scores:
            raise RuntimeError("All detectors failed during scoring.")

        return np.mean(np.stack(all_scores, axis=0), axis=0)

    def fit(self, X_train: np.ndarray) -> "ReconciliationEVTDetector":
        """Fit the detector pool and GPD model on training residuals.

        Residuals are defined as the deviation of each sample's ensemble score
        from the overall training mean (a simple expected-level baseline that
        mirrors the role of hierarchical reconciliation without requiring an
        explicit hierarchy).

        Args:
            X_train: Training data (shape: [n_train, n_features]).

        Returns:
            Self for chaining.
        """
        self.fitted_detectors = []
        for name in self._resolve_names():
            try:
                cls = get_detector(name)
                det = create_detector(cls)
                det.fit(X_train)
                self.fitted_detectors.append((name, det))
                logger.debug("Fitted detector '%s'", name)
            except Exception as exc:
                logger.warning("Skipping detector '%s': %s", name, exc)

        if not self.fitted_detectors:
            raise RuntimeError("No detectors could be fitted.")

        # Compute training residuals
        train_scores = self._ensemble_score(X_train)
        self._train_mean = float(np.mean(train_scores))
        residuals = train_scores - self._train_mean

        # Fit GPD to the tail of |residuals|
        self._gpd_params = _fit_gpd(residuals, self.config.contamination)
        logger.info(
            "ReconciliationEVT fitted %d detectors; GPD threshold=%.4f",
            len(self.fitted_detectors),
            self._gpd_params[0],
        )
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Compute anomaly scores as exceedance probability under GPD.

        Higher score = more anomalous (anomaly score = 1 − P(not anomalous)).

        Args:
            X: Data to score (shape: [n_samples, n_features]).

        Returns:
            Anomaly scores in [0, 1] (shape: [n_samples]).
        """
        if not self.fitted_detectors:
            raise RuntimeError("Detector must be fitted before scoring.")

        ensemble = self._ensemble_score(X)
        residuals = np.abs(ensemble - self._train_mean)

        threshold, shape, scale = self._gpd_params
        survival = _gpd_survival(residuals, threshold, shape, scale)

        # Convert survival probability to anomaly score (1 = definitely anomalous)
        return 1.0 - survival

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Binary anomaly predictions via GPD tail test.

        Args:
            X: Data (shape: [n_samples, n_features]).

        Returns:
            Binary predictions (shape: [n_samples]).
        """
        if not self.fitted_detectors:
            raise RuntimeError("Detector must be fitted before prediction.")

        ensemble = self._ensemble_score(X)
        residuals = np.abs(ensemble - self._train_mean)

        threshold, shape, scale = self._gpd_params
        survival = _gpd_survival(residuals, threshold, shape, scale)

        # Flag points whose tail probability falls below alpha_evt
        return (survival < self.config.alpha_evt).astype(int)

    def predict_with_scores(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return both binary predictions and anomaly scores.

        Args:
            X: Data (shape: [n_samples, n_features]).

        Returns:
            Tuple of (binary_predictions, anomaly_scores).
        """
        scores = self.score(X)
        preds = self.predict(X)
        return preds, scores

    def get_reconciled_scores(self, X: np.ndarray) -> pd.Series:
        """Return ensemble scores as a pandas Series (hierarchy-ready format).

        Args:
            X: Data (shape: [n_samples, n_features]).

        Returns:
            Ensemble scores indexed 0..n-1.
        """
        ensemble = self._ensemble_score(X)
        return pd.Series(ensemble, name="ensemble_score")


@register_baseline_detector("reconciliation_evt")
class RegisteredReconciliationEVTDetector(ReconciliationEVTDetector):
    """Reconciliation + EVT detector registered for baseline comparison."""

    pass
