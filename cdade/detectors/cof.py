"""Connectivity-Based Outlier Factor outlier detector.

Uses PyOD COF implementation.
"""

from dataclasses import dataclass

import numpy as np
from pyod.models.cof import COF as PyODCOF

from cdade.registry import register_detector


@dataclass(frozen=True)
class COFDetectorConfig:
    """COF detector configuration.

    Args:
    """

    # COF doesn't have many tunable parameters
    contamination: float = 0.05


@register_detector("cof")
class COFDetector:
    """Connectivity-Based Outlier Factor outlier detection.

    Uses PyOD COF implementation.

    Args:
        cfg: COFDetectorConfig instance
    """

    def __init__(self, cfg: COFDetectorConfig):
        self.cfg = cfg
        self.model = PyODCOF(contamination=cfg.contamination)

    def fit(self, X: "np.ndarray") -> "COFDetector":
        """Fit detector on training data.

        Args:
            X: Training data of shape (n_samples, n_features)

        Returns:
            Self for chaining
        """
        self.model.fit(X)
        return self

    def score(self, X: "np.ndarray") -> "np.ndarray":
        """Return anomaly scores (higher = more anomalous).

        Args:
            X: Data to score

        Returns:
            Anomaly scores of shape (n_samples,)
        """
        return self.model.decision_function(X)
