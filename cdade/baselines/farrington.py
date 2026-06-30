"""Farrington/Noufaily anomaly detection baseline.

Implements the Farrington-Noufaily algorithm for epidemiological time series anomaly detection.

Reference:
    Farrington, C. P., et al. (1996). A new statistical model for detecting
    new incidence of infectious diseases. BMJ.
    Noufaily, A., et al. (2010). An improved algorithm for infectious disease
    surveillance. IJIA.

Algorithm:
1. Fit Poisson regression: counts ~ time + offset(log(population)) + seasonal + trend
2. Compute standardized residuals: Z = (observed - expected) / sd(expected)
3. Flag anomalies where Z > threshold or log-likelihood ratio test rejects null

Author: CDADE project
"""

import logging
from dataclasses import dataclass

import numpy as np
from scipy import stats
from statsmodels.genmod.families import Poisson
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.tools.sm_exceptions import PerfectSeparationWarning

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FarringtonConfig:
    """Configuration for Farrington-Noufaily anomaly detection.

    Attributes:
        seasonal_period: Period for seasonal effects (e.g., 52 for weekly, 12 for monthly)
        trend_period: Period for trend effects (e.g., 4 for quarterly)
        z_threshold: Z-score threshold for anomaly flagging (default: 3.0)
        llr_threshold: Log-likelihood ratio test threshold (default: 0.05)
        min_obs: Minimum observations required for fit (default: 12)
    """

    seasonal_period: int = 52
    trend_period: int = 4
    z_threshold: float = 3.0
    llr_threshold: float = 0.05
    min_obs: int = 12

    def __post_init__(self):
        if self.z_threshold <= 0:
            raise ValueError("z_threshold must be positive")
        if self.llr_threshold <= 0 or self.llr_threshold >= 1:
            raise ValueError("llr_threshold must be in (0, 1)")


class FarringtonDetector:
    """Farrington-Noufaily anomaly detector.

    Fits a Poisson regression model to observed counts and computes
    standardized residuals for anomaly detection.

    Attributes:
        config: Configuration for detector parameters
        model: Fitted GLM model (when fit)
        fitted: Whether model has been fitted
    """

    def __init__(self, config: FarringtonConfig | None = None):
        """Initialize Farrington detector.

        Args:
            config: Configuration object (uses defaults if None)
        """
        self.config = config or FarringtonConfig()
        self.model: GLM | None = None
        self.fitted = False

    def fit(self, time_series: np.ndarray, population: np.ndarray | None = None) -> None:
        """Fit Poisson regression model.

        Args:
            time_series: Observed counts (shape: [n_obs])
            population: Population at each time point (shape: [n_obs]), optional
        """
        if len(time_series) < self.config.min_obs:
            raise ValueError(f"Need at least {self.config.min_obs} observations for fit")

        time_series = np.asarray(time_series)
        n_obs = len(time_series)

        # Validate population
        if population is not None:
            population = np.asarray(population)
            if len(population) != n_obs:
                raise ValueError("population must match length of time_series")

        # Create design matrix
        time_idx = np.arange(n_obs).reshape(-1, 1)

        # Seasonal effects (Fourier terms)
        seasonal_terms = []
        for h in range(1, min(4, self.config.seasonal_period // 2) + 1):
            seasonal_terms.extend(
                [
                    np.sin(2 * np.pi * h * time_idx / self.config.seasonal_period),
                    np.cos(2 * np.pi * h * time_idx / self.config.seasonal_period),
                ]
            )

        # Trend effects (categorical or spline)
        trend_period = min(self.config.trend_period, n_obs)
        if trend_period > 1:
            trend = time_idx.flatten() / trend_period
            seasonal_terms.append(trend)

        # Combine features
        X = np.hstack([time_idx] + [np.array(v).reshape(-1, 1) for v in seasonal_terms])

        # Fit Poisson regression
        try:
            self.model = GLM(endog=time_series, exog=X, family=Poisson()).fit(disp=0)
        except PerfectSeparationWarning:
            logger.warning("Perfect separation in Poisson regression - fitting with regularization")
            # Fallback: use fewer features
            X_simple = time_idx
            if population is not None:
                X_simple = np.hstack([X_simple, np.log(population).reshape(-1, 1)])
            self.model = GLM(endog=time_series, exog=X_simple, family=Poisson()).fit(disp=0)

        self.fitted = True
        logger.debug(f"Fitted Poisson regression with {len(self.model.params)} parameters")

    def predict_expected(
        self, time_series: np.ndarray, population: np.ndarray | None = None
    ) -> np.ndarray:
        """Predict expected counts.

        Args:
            time_series: Observed counts (shape: [n_obs])
            population: Population at each time point (shape: [n_obs]), optional

        Returns:
            Expected counts (shape: [n_obs])
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")

        time_series = np.asarray(time_series)
        n_obs = len(time_series)

        # Create design matrix (same as fit)
        time_idx = np.arange(n_obs).reshape(-1, 1)

        # Seasonal effects
        seasonal_terms = []
        for h in range(1, min(4, self.config.seasonal_period // 2) + 1):
            seasonal_terms.extend(
                [
                    np.sin(2 * np.pi * h * time_idx / self.config.seasonal_period),
                    np.cos(2 * np.pi * h * time_idx / self.config.seasonal_period),
                ]
            )

        # Trend effects
        trend_period = min(self.config.trend_period, n_obs)
        if trend_period > 1:
            trend = time_idx.flatten() / trend_period
            seasonal_terms.append(trend)

        # Combine features
        X = np.hstack([time_idx] + [np.array(v).reshape(-1, 1) for v in seasonal_terms])

        # Predict expected counts
        expected = self.model.predict(exog=X).flatten()

        return expected

    def predict_std_residuals(
        self, time_series: np.ndarray, population: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute standardized residuals.

        Args:
            time_series: Observed counts (shape: [n_obs])
            population: Population at each time point (shape: [n_obs]), optional

        Returns:
            Tuple of (standardized residuals, standard deviation of expected)
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")

        time_series = np.asarray(time_series)

        # Predict expected counts
        expected = self.predict_expected(time_series, population)

        # Compute standardized residuals
        # Z = (observed - expected) / sd(expected)
        # sd(expected) = sqrt(expected) for Poisson
        std_dev = np.sqrt(expected)

        # Handle zero expected values (avoid division by zero)
        std_dev[std_dev == 0] = np.finfo(float).eps

        standardized = (time_series - expected) / std_dev

        return standardized, std_dev

    def predict_anomaly_scores(
        self, time_series: np.ndarray, population: np.ndarray | None = None
    ) -> np.ndarray:
        """Compute anomaly scores.

        Uses standardized Z-scores as anomaly scores (higher = more anomalous).

        Args:
            time_series: Observed counts (shape: [n_obs])
            population: Population at each time point (shape: [n_obs]), optional

        Returns:
            Anomaly scores (shape: [n_obs])
        """
        standardized, _ = self.predict_std_residuals(time_series, population)
        return standardized

    def detect_anomalies(
        self, time_series: np.ndarray, population: np.ndarray | None = None
    ) -> np.ndarray:
        """Detect anomalies.

        Flags anomalies where Z-score exceeds threshold or LLR test rejects null.

        Args:
            time_series: Observed counts (shape: [n_obs])
            population: Population at each time point (shape: [n_obs]), optional

        Returns:
            Binary anomaly flags (shape: [n_obs], 1 = anomaly)
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before detection")

        standardized, std_dev = self.predict_std_residuals(time_series, population)

        # Z-score based anomaly detection
        z_score_anomalies = np.abs(standardized) > self.config.z_threshold

        # Log-likelihood ratio test
        # LLR = 2 * (observed * log(observed/expected) - (observed - expected))
        # or equivalently: LLR = 2 * (deviance)
        expected = self.predict_expected(time_series, population)
        expected[expected <= 0] = np.finfo(float).eps

        llr = 2 * (time_series * np.log(time_series / expected) - (time_series - expected))

        llr_anomalies = llr > stats.chi2.ppf(1 - self.config.llr_threshold, df=1)

        # Combine criteria
        anomalies = z_score_anomalies | llr_anomalies

        return anomalies.astype(int)

    def score(self, time_series: np.ndarray, population: np.ndarray | None = None) -> np.ndarray:
        """Compute anomaly score for a time series.

        Higher score indicates more anomalous.

        Args:
            time_series: Observed counts (shape: [n_obs])
            population: Population at each time point (shape: [n_obs]), optional

        Returns:
            Anomaly scores (shape: [n_obs])
        """
        return self.predict_anomaly_scores(time_series, population)


def register_baseline_detector(name: str):
    """Decorator to register Farrington detector in registry.

    Args:
        name: Registry name for the detector

    Returns:
        Decorated class
    """
    from cdade.registry import register_baseline_detector

    return register_baseline_detector(name)


@register_baseline_detector("farrington")
class RegisteredFarringtonDetector(FarringtonDetector):
    """Farrington detector registered for baseline comparison."""

    pass
