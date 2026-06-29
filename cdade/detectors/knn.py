"""k-Nearest Neighbors outlier detector.

Uses PyOD KNN implementation.
"""

from dataclasses import dataclass

import numpy as np
from pyod.models.knn import KNN as PyODKNN

from cdade.registry import register_detector


@dataclass(frozen=True)
class KNNDetectorConfig:
    """KNN detector configuration.

    Args:
        n_neighbors: Number of neighbors to use
        contamination: Expected proportion of outliers
        random_state: Seed for reproducibility
    """

    n_neighbors: int = 20
    contamination: float = 0.05
    random_state: int = 42


@register_detector("knn")
class KNNDetector:
    """k-Nearest Neighbors outlier detection.

    Uses PyOD KNN implementation.

    Args:
        cfg: KNNDetectorConfig instance
    """

    def __init__(self, cfg: KNNDetectorConfig):
        self.cfg = cfg
        self.model = PyODKNN(
            n_neighbors=cfg.n_neighbors,
            contamination=cfg.contamination,
            random_state=cfg.random_state,
        )

    def fit(self, X: "np.ndarray") -> "KNNDetector":
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
