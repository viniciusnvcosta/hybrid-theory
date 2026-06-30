"""Best single detector baseline for CDADE comparison.

Implements baseline that selects the best performing detector from the detector pool
based on validation performance and applies it to test data.

Reference:
    Eze, J. et al. (2023). Anomaly Detection in Hierarchical Time Series.
    Uses validation set to select best detector.

Author: CDADE project
"""

import logging
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

from cdade.registry import get_detector, list_detectors

logger = logging.getLogger(__name__)


def create_detector(detector_cls):
    """Create a detector instance with appropriate config.

    Args:
        detector_cls: Detector class (may use config pattern).

    Returns:
        Detector instance.
    """
    from dataclasses import MISSING, fields

    # For detectors that use config pattern - they take 'cfg' as first parameter
    try:
        # Get the type of the cfg parameter
        import inspect

        sig = inspect.signature(detector_cls.__init__)
        if "cfg" in sig.parameters:
            cfg_param = sig.parameters["cfg"]
            # Check if cfg is a dataclass type
            cfg_type = cfg_param.annotation
            if hasattr(cfg_type, "__dataclass_fields__"):
                # Create config instance with defaults
                config_dict = {}
                for field in fields(cfg_type):
                    if field.default is not MISSING:
                        config_dict[field.name] = field.default
                    elif field.default_factory is not MISSING:
                        config_dict[field.name] = field.default_factory()
                # Create config object
                cfg_instance = cfg_type(**config_dict)
                # Instantiate detector with cfg object
                return detector_cls(cfg=cfg_instance)
    except Exception as e:
        logger.debug(f"create_detector exception: {type(e).__name__}: {e}")

    # Use default config - instantiate without args
    return detector_cls()


@dataclass
class SimpleDetectorConfig:
    """Simple detector configuration for testing."""

    contamination: float = 0.05


def evaluate_detector(
    detector_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, float]:
    """Evaluate a detector on train and test sets.

    Args:
        detector_name: Name of the detector to evaluate.
        X_train: Training data (shape: [n_train, n_features]).
        y_train: Training labels (shape: [n_train]).
        X_test: Test data (shape: [n_test, n_features]).
        y_test: Test labels (shape: [n_test]).

    Returns:
        Dictionary of metrics: precision, recall, f1.
    """
    try:
        # Get detector class
        detector_cls = get_detector(detector_name)

        # For detectors that use config pattern, we need to create config
        if (
            hasattr(detector_cls, "__dataclass_fields__")
            and "cfg" in detector_cls.__dataclass_fields__
        ):
            # Use a default config - extract from type hints or use defaults
            try:
                import typing
                from dataclasses import MISSING, fields

                # Try to get type hints for cfg
                cfg_type_hints = typing.get_type_hints(detector_cls.__init__)
                if "cfg" in cfg_type_hints:
                    cfg_type = cfg_type_hints["cfg"]
                    # Extract config fields from the type
                    if hasattr(cfg_type, "__dataclass_fields__"):
                        config_dict = {}
                        for field in fields(cfg_type):
                            if field.default is not MISSING:
                                config_dict[field.name] = field.default
                            elif field.default_factory is not MISSING:
                                config_dict[field.name] = field.default_factory()
                        detector = detector_cls(**config_dict)
                    else:
                        # Use default values from dataclass
                        detector = detector_cls()
                else:
                    # Use default values from dataclass
                    detector = detector_cls()
            except Exception as cfg_error:
                # Fallback: use default config
                try:
                    from dataclasses import MISSING, fields

                    for field in fields(detector_cls):
                        if field.default is not MISSING:
                            detector = detector_cls(**{field.name: field.default})
                            break
                    else:
                        detector = detector_cls()
                except Exception:
                    # Last resort: raise error
                    raise cfg_error from None
        else:
            # Standard detector without config
            detector = detector_cls()

        # Fit on training data
        detector.fit(X_train)

        # Predict on training data
        train_scores = detector.score(X_train)

        # Predict on test data
        test_scores = detector.score(X_test)

        # Threshold scores to binary predictions
        # Using median of train scores as threshold
        threshold = np.median(train_scores)
        test_pred = (test_scores >= threshold).astype(int)

        # Compute metrics (ignore labels)
        precision = precision_score(y_test, test_pred, zero_division=0)
        recall = recall_score(y_test, test_pred, zero_division=0)
        f1 = f1_score(y_test, test_pred, zero_division=0)

        return {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "threshold": float(threshold),
        }

    except Exception as e:
        logger.warning(f"Detector {detector_name} failed: {e}")
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "threshold": 0.0}


def find_best_detector(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    detector_names: list[str] | None = None,
) -> tuple[str, dict[str, float]]:
    """Find the best detector based on F1 score on validation set.

    Args:
        X_train: Training data.
        y_train: Training labels.
        X_test: Validation/test data.
        y_test: Validation/test labels.
        detector_names: List of detector names to evaluate. If None, uses all registered detectors.

    Returns:
        Tuple of (best_detector_name, metrics_dict).
    """
    if detector_names is None:
        detector_names = list_detectors()

    best_detector = None
    best_f1 = -1.0
    best_metrics = {}

    for detector_name in detector_names:
        metrics = evaluate_detector(detector_name, X_train, y_train, X_test, y_test)

        f1 = metrics["f1"]
        if f1 > best_f1:
            best_f1 = f1
            best_detector = detector_name
            best_metrics = metrics

    logger.info(f"Best detector: {best_detector} (F1={best_f1:.4f})")
    return best_detector, best_metrics


class BestSingleDetector:
    """Best single detector baseline.

    Selects the best detector from the detector pool based on validation performance
    and applies it to test data.

    Attributes:
        best_detector_name: Name of the best detector.
        best_metrics: Metrics for the best detector.
    """

    def __init__(self, detector_names: list[str] | None = None):
        """Initialize best single detector.

        Args:
            detector_names: List of detector names to evaluate. If None, uses all
                registered detectors.
        """
        self.detector_names = detector_names
        self.best_detector_name: str | None = None
        self.best_metrics: dict[str, float] = {}

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Fit the detector by finding the best one on validation data.

        Args:
            X_train: Training data (shape: [n_train, n_features]).
            y_train: Training labels (shape: [n_train]).
        """
        # Split data into train and validation
        n_samples = len(X_train)
        n_train = int(0.8 * n_samples)

        X_val = X_train[n_train:]
        y_val = y_train[n_train:]

        X_train_split = X_train[:n_train]
        y_train_split = y_train[:n_train]

        # Find best detector
        self.best_detector_name, self.best_metrics = find_best_detector(
            X_train_split, y_train_split, X_val, y_val, self.detector_names
        )

        logger.info(f"Selected best detector: {self.best_detector_name}")
        logger.info(
            f"Validation metrics: F1={self.best_metrics['f1']:.4f}, "
            f"Precision={self.best_metrics['precision']:.4f}, "
            f"Recall={self.best_metrics['recall']:.4f}"
        )

    def score(self, X_test: np.ndarray) -> np.ndarray:
        """Compute anomaly scores using the best detector.

        Args:
            X_test: Test data (shape: [n_test, n_features]).

        Returns:
            Anomaly scores (shape: [n_test]).
        """
        if self.best_detector_name is None:
            raise ValueError("Detector must be fitted before scoring")

        try:
            # Get detector class
            detector_cls = get_detector(self.best_detector_name)

            # For detectors that use config pattern
            detector = create_detector(detector_cls)

            # Fit and score
            detector.fit(X_test)
            return detector.score(X_test)

        except Exception as e:
            logger.error(f"Failed to score with detector {self.best_detector_name}: {e}")
            raise

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Predict binary anomaly flags.

        Args:
            X_test: Test data (shape: [n_test, n_features]).

        Returns:
            Binary predictions (shape: [n_test]).
        """
        scores = self.score(X_test)
        threshold = self.best_metrics.get("threshold", np.median(scores))
        return (scores >= threshold).astype(int)


def register_baseline_detector(name: str):
    """Decorator to register best single detector in registry.

    Args:
        name: Registry name for the detector

    Returns:
        Decorated class
    """
    from cdade.registry import register_baseline_detector

    return register_baseline_detector(name)


@register_baseline_detector("single_best")
class RegisteredBestSingleDetector(BestSingleDetector):
    """Best single detector registered for baseline comparison."""

    pass
