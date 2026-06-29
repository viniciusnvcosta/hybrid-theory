"""Principal Component Analysis outlier detector.

Uses PyOD PCA implementation.
"""

from dataclasses import dataclass

import numpy as np
from pyod.models.pca import PCA as PyODPCA

from cdade.registry import register_detector


@dataclass(frozen=True)
class PCADetectorConfig:
    """PCA detector configuration.

    Args:
        n_components: Number of components to keep
        contamination: Expected proportion of outliers
        random_state: Seed for reproducibility
    """

    n_components: float = 0.95
    contamination: float = 0.05
    random_state: int = 42


@register_detector("pca")
class PCADetector:
    """Principal Component Analysis outlier detection.

    Uses PyOD PCA implementation. Returns negative decision function
    (higher = more anomalous).

    Args:
        cfg: PCADetectorConfig instance
    """

    def __init__(self, cfg: PCADetectorConfig):
        self.cfg = cfg
        self.model = PyODPCA(
            n_components=cfg.n_components,
            contamination=cfg.contamination,
            random_state=cfg.random_state,
        )

    def fit(self, X: "np.ndarray") -> "PCADetector":
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
        # PyOD PCA returns negative decision function → invert for higher = more anomalous
        return -self.model.decision_function(X)
