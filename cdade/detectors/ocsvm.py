"""One-Class SVM outlier detector.

Uses PyOD OCSVM implementation.
"""

from dataclasses import dataclass

import numpy as np
from pyod.models.ocsvm import OCSVM as PyODOCSVM

from cdade.registry import register_detector


@dataclass(frozen=True)
class OCSVMDetectorConfig:
    """OCSVM detector configuration.

    Args:
        nu: Upper bound on fraction of training errors
        contamination: Expected proportion of outliers
        random_state: Seed for reproducibility
    """

    nu: float = 0.1
    contamination: float = 0.05
    random_state: int = 42


@register_detector("ocsvm")
class OCSVMDetector:
    """One-Class SVM outlier detection.

    Uses PyOD OCSVM implementation.

    Args:
        cfg: OCSVMDetectorConfig instance
    """

    def __init__(self, cfg: OCSVMDetectorConfig):
        self.cfg = cfg
        self.model = PyODOCSVM(nu=cfg.nu, contamination=cfg.contamination)

    def fit(self, X: "np.ndarray") -> "OCSVMDetector":
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
