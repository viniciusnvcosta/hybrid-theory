"""Tests for B2: Best Single Detector baseline.

Tests the best single detector selection and scoring.

Author: CDADE project
"""

import numpy as np
import pytest

from cdade.baselines.single_best import (
    BestSingleDetector,
    evaluate_detector,
    find_best_detector,
)

# Import detectors first to ensure they're registered
from cdade.detectors import iforest, lof, mcd, pca  # noqa: F401
from cdade.registry import list_detectors


class TestFindBestDetector:
    """Test detector selection logic."""

    def test_find_best_detector_simple(self):
        """Test finding best detector on synthetic data."""
        # Create synthetic data
        np.random.seed(42)
        n_train = 100
        n_test = 50

        X_train = np.random.randn(n_train, 10)
        y_train = np.random.randint(0, 2, n_train)

        X_test = np.random.randn(n_test, 10)
        y_test = np.random.randint(0, 2, n_test)

        # Check what detectors are available
        available_detectors = list_detectors()
        print(f"Available detectors: {available_detectors}")

        # Find best detector
        best_detector, metrics = find_best_detector(X_train, y_train, X_test, y_test)

        # Debug output
        print(f"Best detector: {best_detector}")
        print(f"Metrics: {metrics}")

        # Should return a detector
        assert best_detector is not None
        assert isinstance(best_detector, str)

        # Should have metrics
        assert "f1" in metrics
        assert "precision" in metrics
        assert "recall" in metrics

        # F1 should be between 0 and 1
        assert 0.0 <= metrics["f1"] <= 1.0

    def test_find_best_detector_with_specific_detectors(self):
        """Test finding best detector from specific list."""
        np.random.seed(42)
        X_train = np.random.randn(100, 10)
        y_train = np.random.randint(0, 2, 100)
        X_test = np.random.randn(50, 10)
        y_test = np.random.randint(0, 2, 50)

        # Find best from specific list
        detectors_to_test = ["pca", "iforest"]
        best_detector, metrics = find_best_detector(
            X_train, y_train, X_test, y_test, detector_names=detectors_to_test
        )

        # Should return one of the specified detectors
        assert best_detector in detectors_to_test

    def test_find_best_detector_all_detectors(self):
        """Test finding best from all detectors."""
        np.random.seed(42)
        X_train = np.random.randn(100, 10)
        y_train = np.random.randint(0, 2, 100)
        X_test = np.random.randn(50, 10)
        y_test = np.random.randint(0, 2, 50)

        # Find best from all detectors
        best_detector, metrics = find_best_detector(X_train, y_train, X_test, y_test)

        # Should successfully find a detector
        assert best_detector is not None
        assert metrics["f1"] >= 0.0


class TestEvaluateDetector:
    """Test detector evaluation."""

    def test_evaluate_detector_basic(self):
        """Test basic detector evaluation."""
        np.random.seed(42)
        X_train = np.random.randn(100, 10)
        y_train = np.random.randint(0, 2, 100)
        X_test = np.random.randn(50, 10)
        y_test = np.random.randint(0, 2, 50)

        metrics = evaluate_detector("pca", X_train, y_train, X_test, y_test)

        # Should return metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert "threshold" in metrics

        # Should be non-negative
        assert metrics["precision"] >= 0.0
        assert metrics["recall"] >= 0.0
        assert metrics["f1"] >= 0.0

    def test_evaluate_detector_all_zeros(self):
        """Test evaluation with all zero metrics."""
        metrics = evaluate_detector(
            "nonexistent_detector",
            np.random.randn(100, 10),
            np.random.randint(0, 2, 100),
            np.random.randn(50, 10),
            np.random.randint(0, 2, 50),
        )

        # Should return zeros for invalid detector
        assert metrics["precision"] == 0.0
        assert metrics["recall"] == 0.0
        assert metrics["f1"] == 0.0
        assert metrics["threshold"] == 0.0


class TestBestSingleDetector:
    """Test BestSingleDetector class."""

    def test_fit_and_score(self):
        """Test fitting and scoring."""
        np.random.seed(42)
        n_samples = 120
        X_train = np.random.randn(n_samples, 10)
        y_train = np.random.randint(0, 2, n_samples)

        detector = BestSingleDetector()
        detector.fit(X_train, y_train)

        # Should have selected a detector
        assert detector.best_detector_name is not None
        assert detector.best_metrics is not None

        # Should be able to score
        scores = detector.score(np.random.randn(50, 10))
        assert len(scores) == 50
        assert np.all(np.isfinite(scores))

    def test_fit_and_predict(self):
        """Test fitting and predicting."""
        np.random.seed(42)
        n_samples = 120
        X_train = np.random.randn(n_samples, 10)
        y_train = np.random.randint(0, 2, n_samples)

        detector = BestSingleDetector()
        detector.fit(X_train, y_train)

        # Should be able to predict
        predictions = detector.predict(np.random.randn(50, 10))
        assert len(predictions) == 50
        assert np.all(predictions >= 0)
        assert np.all(predictions <= 1)

    def test_fit_before_score(self):
        """Test scoring before fit raises error."""
        detector = BestSingleDetector()

        with pytest.raises(ValueError, match="Detector must be fitted before scoring"):
            detector.score(np.random.randn(50, 10))

    def test_fit_before_predict(self):
        """Test predicting before fit raises error."""
        detector = BestSingleDetector()

        with pytest.raises(ValueError, match="Detector must be fitted before scoring"):
            detector.predict(np.random.randn(50, 10))

    def test_custom_detector_names(self):
        """Test with custom detector names."""
        np.random.seed(42)
        X_train = np.random.randn(100, 10)
        y_train = np.random.randint(0, 2, 100)

        detector = BestSingleDetector(detector_names=["pca", "iforest"])
        detector.fit(X_train, y_train)

        # Should have selected a detector
        assert detector.best_detector_name in ["pca", "iforest"]

    def test_fit_splits_data(self):
        """Test that fit splits data into train/val."""
        np.random.seed(42)
        n_samples = 120
        X_train = np.random.randn(n_samples, 10)
        y_train = np.random.randint(0, 2, n_samples)

        detector = BestSingleDetector()
        detector.fit(X_train, y_train)

        # Should have selected a detector
        assert detector.best_detector_name is not None

        # Should have validation metrics
        assert detector.best_metrics is not None
        assert detector.best_metrics["f1"] >= 0.0


class TestBestSingleDetectorRegistry:
    """Test registry integration."""

    def test_registered_detector_available(self):
        """Test that detector is registered."""
        from cdade.registry import list_baseline_detectors

        assert "single_best" in list_baseline_detectors()
        assert len(list_baseline_detectors()) >= 1
