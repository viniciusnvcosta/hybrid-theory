"""Clustering-Based Local Outlier Factor outlier detector.

Uses PyOD CBLOF implementation.
"""

from dataclasses import dataclass

import numpy as np
from pyod.models.cblof import CBLOF as PyODCBLOF

from cdade.registry import register_detector


@dataclass(frozen=True)
class CBLOFDetectorConfig:
    """CBLOF detector configuration.

    Args:
        n_clusters: Number of clusters
        contamination: Expected proportion of outliers
        random_state: Seed for reproducibility
    """

    n_clusters: int = 10
    contamination: float = 0.05
    random_state: int = 42


@register_detector("cblof")
class CBLOFDetector:
    """Clustering-Based Local Outlier Factor outlier detection.

    Uses PyOD CBLOF implementation.

    Args:
        cfg: CBLOFDetectorConfig instance
    """

    def __init__(self, cfg: CBLOFDetectorConfig):
        self.cfg = cfg
        self.model = PyODCBLOF(
            n_clusters=cfg.n_clusters,
            contamination=cfg.contamination,
            random_state=cfg.random_state,
        )

    def fit(self, X: "np.ndarray") -> "CBLOFDetector":
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
