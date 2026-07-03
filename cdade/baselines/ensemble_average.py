"""Full ensemble average baseline (B3) for CDADE comparison.

Implements the Average of Maximum (AOM) strategy: fits all registered detectors
and averages their anomaly scores as the ensemble output.

Reference:
    Aggarwal, C. C. (2013). Outlier Analysis. Springer.
    AOM: Average of Maximum combination strategy.

Author: CDADE project
"""

import logging
from dataclasses import dataclass, field

import numpy as np

from cdade.baselines.single_best import create_detector
from cdade.registry import get_detector, list_detectors, register_baseline_detector

logger = logging.getLogger(__name__)


@dataclass
class EnsembleAverageConfig:
    """Configuration for the full ensemble average baseline.

    Attributes:
        detector_names: Names of detectors to include. Uses all registered
            detectors when empty.
        normalize: Whether to normalise each detector's scores to [0, 1] before
            averaging. Prevents detectors with large score ranges from dominating.
    """

    detector_names: list[str] = field(default_factory=list)
    normalize: bool = True


def _normalize_scores(scores: np.ndarray) -> np.ndarray:
    """Normalise scores to [0, 1] via min-max scaling.

    Args:
        scores: Raw anomaly scores (shape: [n_samples]).

    Returns:
        Normalised scores in [0, 1].
    """
    lo, hi = scores.min(), scores.max()
    if hi == lo:
        return np.zeros_like(scores)
    return (scores - lo) / (hi - lo)


class EnsembleAverageDetector:
    """Full ensemble average detector (AOM baseline).

    Fits every detector in the pool on the training data, then averages
    (optionally normalised) scores at inference time.

    Attributes:
        config: Detector configuration.
        fitted_detectors: List of (name, detector) pairs after fitting.
    """

    def __init__(self, config: EnsembleAverageConfig | None = None) -> None:
        """Initialise the ensemble average detector.

        Args:
            config: Configuration object; uses defaults when None.
        """
        self.config = config or EnsembleAverageConfig()
        self.fitted_detectors: list[tuple[str, object]] = []

    def _resolve_names(self) -> list[str]:
        return self.config.detector_names if self.config.detector_names else list_detectors()

    def fit(self, X_train: np.ndarray) -> "EnsembleAverageDetector":
        """Fit all detectors on training data.

        Args:
            X_train: Training data (shape: [n_samples, n_features]).

        Returns:
            Self for chaining.
        """
        self.fitted_detectors = []
        names = self._resolve_names()

        for name in names:
            try:
                cls = get_detector(name)
                detector = create_detector(cls)
                detector.fit(X_train)
                self.fitted_detectors.append((name, detector))
                logger.debug("Fitted detector '%s'", name)
            except Exception as exc:
                logger.warning("Skipping detector '%s': %s", name, exc)

        if not self.fitted_detectors:
            raise RuntimeError("No detectors could be fitted.")

        logger.info("EnsembleAverage fitted %d detectors.", len(self.fitted_detectors))
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Compute averaged anomaly scores.

        Args:
            X: Data to score (shape: [n_samples, n_features]).

        Returns:
            Averaged anomaly scores (shape: [n_samples]).
        """
        if not self.fitted_detectors:
            raise RuntimeError("Detector must be fitted before scoring.")

        all_scores: list[np.ndarray] = []
        for name, detector in self.fitted_detectors:
            try:
                s = detector.score(X)
                if self.config.normalize:
                    s = _normalize_scores(s)
                all_scores.append(s)
            except Exception as exc:
                logger.warning("Detector '%s' failed at score time: %s", name, exc)

        if not all_scores:
            raise RuntimeError("All detectors failed during scoring.")

        return np.mean(np.stack(all_scores, axis=0), axis=0)

    def predict(self, X: np.ndarray, threshold: float | None = None) -> np.ndarray:
        """Binary anomaly predictions via score thresholding.

        Args:
            X: Data (shape: [n_samples, n_features]).
            threshold: Score threshold. Uses median of scores when None.

        Returns:
            Binary predictions (shape: [n_samples]).
        """
        scores = self.score(X)
        thresh = threshold if threshold is not None else float(np.median(scores))
        return (scores >= thresh).astype(int)


@register_baseline_detector("ensemble_average")
class RegisteredEnsembleAverageDetector(EnsembleAverageDetector):
    """Full ensemble average detector registered for baseline comparison."""

    pass
