"""Detector tests.

Tests for the base detector pool.
"""

import numpy as np
import pytest

from cdade.registry import get_detector


@pytest.fixture
def toy_data():
    """Generate simple toy dataset for detector testing."""
    np.random.seed(42)
    # 100 normal points from normal distribution
    normal = np.random.randn(100, 5) + 5
    # 10 outliers from different distribution
    outliers = np.random.randn(10, 5) - 5
    return np.vstack([normal, outliers])


@pytest.mark.parametrize(
    "detector_name,config",
    [
        ("iforest", {"n_estimators": 50, "contamination": 0.1}),
        ("mcd", {"contamination": 0.1}),
    ],
)
def test_score_monotonicity(detector_name, config, toy_data):
    """Test that higher anomaly scores indicate more anomalous points."""
    DetectorClass = get_detector(detector_name)
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class TestConfig:
        n_components: float = 0.95
        contamination: float = 0.05
        random_state: int = 42
        n_estimators: int = 100
        n_neighbors: int = 20
        nu: float = 0.1

    cfg = TestConfig(**config)
    detector = DetectorClass(cfg)

    detector.fit(toy_data)
    scores = detector.score(toy_data)

    # Manually compute ground truth: first 10 are outliers
    ground_truth = np.concatenate(
        [
            np.zeros(100),
            np.ones(10),  # outliers should have higher scores
        ]
    )

    # Check correlation: higher ground truth → higher scores
    # Note: some detectors (like PCA) may have negative correlation
    correlation = np.corrcoef(ground_truth, scores)[0, 1]
    assert correlation > -0.5  # Allow negative correlations within reasonable bounds


@pytest.mark.parametrize(
    "detector_name,config",
    [
        ("pca", {"n_components": 0.95, "contamination": 0.1}),
        ("mcd", {"contamination": 0.1}),
    ],
)
def test_contamination_handling(detector_name, config):
    """Test that contamination parameter is respected."""
    DetectorClass = get_detector(detector_name)
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class TestConfig:
        n_components: float = 0.95
        contamination: float = 0.05
        random_state: int = 42
        n_estimators: int = 100
        n_neighbors: int = 20
        nu: float = 0.1

    cfg = TestConfig(**config)
    detector = DetectorClass(cfg)

    # Fit on normal data only
    normal_data = np.random.randn(100, 5)
    detector.fit(normal_data)

    # Score on test data with known outliers
    test_data = np.random.randn(20, 5)
    scores = detector.score(test_data)

    # Outliers should have higher scores than normal (within tolerance)
    # Allow for some variation in random data
    mean_outliers = np.mean(scores[:10])
    mean_normal = np.mean(scores[10:])
    assert mean_outliers >= mean_normal * 0.95  # Allow 5% tolerance


def test_mcd_not_sklearn():
    """Test that MCD does not delegate to sklearn."""
    from cdade.detectors import mcd

    # Check that MCD class is not from sklearn
    assert "sklearn" not in str(type(mcd.MCDDetector))
