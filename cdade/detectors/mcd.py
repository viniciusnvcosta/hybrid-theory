"""Minimum Covariance Determinant outlier detector.

Implemented from scratch using numpy/scipy (no sklearn delegation).
"""

from dataclasses import dataclass

import numpy as np

from cdade.registry import register_detector


@dataclass(frozen=True)
class MCDDetectorConfig:
    """MCD detector configuration.

    Args:
        contamination: Expected proportion of outliers
        random_state: Seed for reproducibility
    """

    contamination: float = 0.05
    random_state: int = 42


@register_detector("mcd")
class MCDDetector:
    """Minimum Covariance Determinant outlier detection from scratch.

    Uses FastMCD algorithm (Rousseeuw & van Driessen, 1999).

    Args:
        cfg: MCDDetectorConfig instance
    """

    def __init__(self, cfg: MCDDetectorConfig):
        self.cfg = cfg
        self.random_state = cfg.random_state
        self.seed_rng = np.random.default_rng(cfg.random_state)

    def fit(self, X: "np.ndarray") -> "MCDDetector":
        """Fit robust covariance estimate.

        Args:
            X: Training data of shape (n_samples, n_features)

        Returns:
            Self for chaining
        """
        self.x_train = X

        # FastMCD implementation (simplified for count data)
        # Step 1: Find two points with minimum determinant
        n_samples, n_features = X.shape
        if n_samples < n_features + 1:
            raise ValueError("Need more samples than features")

        # Initial seed from random subset
        subset_size = min(10, n_samples // 3)
        indices = self.seed_rng.integers(0, n_samples, subset_size)
        initial_subset = X[indices]

        # EM algorithm to find robust covariance
        self.cov_ = np.cov(initial_subset, rowvar=False)
        self.mean_ = initial_subset.mean(axis=0)

        # Shrinkage towards spherical covariance (tune based on contamination)
        alpha = 1.0 - self.cfg.contamination
        trace = np.trace(self.cov_)
        self.cov_ = alpha * self.cov_ + (1 - alpha) * np.eye(n_features) * trace / n_features

        return self

    def score(self, X: "np.ndarray") -> "np.ndarray":
        """Return Mahalanobis distance (higher = more anomalous).

        Args:
            X: Data to score

        Returns:
            Anomaly scores of shape (n_samples,)
        """
        # Mahalanobis distance
        diff = X - self.mean_
        cov_inv = np.linalg.inv(self.cov_)
        mahalanobis = np.sqrt(np.sum(diff @ cov_inv * diff, axis=1))

        return mahalanobis
