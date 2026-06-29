"""Isolation Forest outlier detector.

Uses PyOD IForest implementation.
"""

from dataclasses import dataclass

import numpy as np
from pyod.models.iforest import IForest as PyODIForest

from cdade.registry import register_detector


@dataclass(frozen=True)
class IFDetectorConfig:
    """IF detector configuration.

    Args:
        n_estimators: Number of trees in forest
        contamination: Expected proportion of outliers
        random_state: Seed for reproducibility
    """

    n_estimators: int = 100
    contamination: float = 0.05
    random_state: int = 42


@register_detector("iforest")
class IFDetector:
    """Isolation Forest outlier detection.

    Uses PyOD IForest implementation.

    Args:
        cfg: IFDetectorConfig instance
    """

    def __init__(self, cfg: IFDetectorConfig):
        self.cfg = cfg
        self.model = PyODIForest(
            n_estimators=cfg.n_estimators,
            contamination=cfg.contamination,
            random_state=cfg.random_state,
        )

    def fit(self, X: "np.ndarray") -> "IFDetector":
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
