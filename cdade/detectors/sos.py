"""Subset Outlier Score outlier detector.

Uses PyOD SOS implementation.
"""

from dataclasses import dataclass

import numpy as np
from pyod.models.sos import SOS as PyODSOS

from cdade.registry import register_detector


@dataclass(frozen=True)
class SOSDetectorConfig:
    """SOS detector configuration.

    Args:
        contamination: Expected proportion of outliers
        random_state: Seed for reproducibility
    """

    contamination: float = 0.05
    random_state: int = 42


@register_detector("sos")
class SOSDetector:
    """Subset Outlier Score outlier detection.

    Uses PyOD SOS implementation.

    Args:
        cfg: SOSDetectorConfig instance
    """

    def __init__(self, cfg: SOSDetectorConfig):
        self.cfg = cfg
        self.model = PyODSOS(contamination=cfg.contamination, random_state=cfg.random_state)

    def fit(self, X: "np.ndarray") -> "SOSDetector":
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
