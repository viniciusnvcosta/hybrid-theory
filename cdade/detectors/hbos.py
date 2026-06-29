"""Histogram-Based Outlier Score outlier detector.

Uses PyOD HBOS implementation.
"""

from dataclasses import dataclass

import numpy as np
from pyod.models.hbos import HBOS as PyODHBOS

from cdade.registry import register_detector


@dataclass(frozen=True)
class HBOSDetectorConfig:
    """HBOS detector configuration.

    Args:
        nbins: Number of bins for histogram
        contamination: Expected proportion of outliers
        random_state: Seed for reproducibility
    """

    nbins: int = 10
    contamination: float = 0.05
    random_state: int = 42


@register_detector("hbos")
class HBOSDetector:
    """Histogram-Based Outlier Score outlier detection.

    Uses PyOD HBOS implementation.

    Args:
        cfg: HBOSDetectorConfig instance
    """

    def __init__(self, cfg: HBOSDetectorConfig):
        self.cfg = cfg
        self.model = PyODHBOS(
            n_bins=cfg.nbins, contamination=cfg.contamination, random_state=cfg.random_state
        )

    def fit(self, X: "np.ndarray") -> "HBOSDetector":
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
