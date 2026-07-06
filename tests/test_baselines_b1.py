"""Tests for B1: Farrington/Noufaily baseline.

Tests Poisson regression model fitting, residual computation,
and anomaly detection on synthetic and real data.

Author: CDADE project
"""

import numpy as np
import pytest

from cdade.baselines.farrington import (
    FarringtonConfig,
    FarringtonDetector,
)


class TestFarringtonConfig:
    """Test Farrington configuration validation."""

    def test_default_config(self):
        """Test default configuration values."""
        config = FarringtonConfig()
        assert config.z_threshold == 3.0
        assert config.llr_threshold == 0.05
        assert config.min_obs == 12

    def test_custom_config(self):
        """Test custom configuration."""
        config = FarringtonConfig(z_threshold=4.0, llr_threshold=0.01, min_obs=8)
        assert config.z_threshold == 4.0
        assert config.llr_threshold == 0.01
        assert config.min_obs == 8

    def test_invalid_thresholds(self):
        """Test invalid threshold values raise errors."""
        with pytest.raises(ValueError, match="z_threshold must be positive"):
            FarringtonConfig(z_threshold=0)

        with pytest.raises(ValueError, match="llr_threshold must be in \\(0, 1\\)"):
            FarringtonConfig(llr_threshold=0)
            FarringtonConfig(llr_threshold=1)


class TestFarringtonFit:
    """Test model fitting."""

    def test_fit_with_enough_data(self):
        """Test fit with sufficient observations."""
        detector = FarringtonDetector()
        time_series = np.arange(100) + np.random.poisson(10, 100)

        detector.fit(time_series)
        assert detector.fitted is True
        assert detector.model is not None

    def test_fit_insufficient_data(self):
        """Test fit with insufficient observations raises error."""
        detector = FarringtonDetector()
        time_series = np.arange(5)  # Less than min_obs=12

        with pytest.raises(ValueError, match="Need at least 12 observations"):
            detector.fit(time_series)

    def test_fit_with_population(self):
        """Test fit with population offset."""
        detector = FarringtonDetector()
        time_series = np.arange(100) + np.random.poisson(10, 100)
        population = np.random.uniform(1000, 10000, 100)

        detector.fit(time_series, population)
        assert detector.fitted is True

    def test_fit_population_mismatch(self):
        """Test fit with mismatched population length raises error."""
        detector = FarringtonDetector()
        time_series = np.arange(100) + np.random.poisson(10, 100)
        population = np.random.uniform(1000, 10000, 50)  # Wrong length

        with pytest.raises(ValueError, match="population must match length of time_series"):
            detector.fit(time_series, population)

    def test_fit_perfect_separation(self):
        """Test fit handles perfect separation warning."""
        detector = FarringtonDetector()
        # Create data that causes perfect separation (all counts 0 or 1)
        time_series = np.random.randint(0, 2, 100)
        population = np.ones(100) * 1000

        # Should not raise, just log warning
        detector.fit(time_series, population)
        assert detector.fitted is True


class TestFarringtonPrediction:
    """Test prediction methods."""

    def test_predict_expected(self):
        """Test expected count prediction."""
        detector = FarringtonDetector()
        time_series = np.arange(100) + np.random.poisson(10, 100)

        detector.fit(time_series)
        expected = detector.predict_expected(time_series)

        assert len(expected) == len(time_series)
        assert np.all(expected >= 0)  # Expected counts should be non-negative

    def test_predict_expected_different_length(self):
        """Test prediction with different time series length."""
        detector = FarringtonDetector()
        time_series = np.arange(100) + np.random.poisson(10, 100)

        detector.fit(time_series)
        expected_new = detector.predict_expected(np.arange(50))

        assert len(expected_new) == 50

    def test_predict_std_residuals(self):
        """Test standardized residual computation."""
        detector = FarringtonDetector()
        time_series = np.arange(100) + np.random.poisson(10, 100)

        detector.fit(time_series)
        residuals, std_dev = detector.predict_std_residuals(time_series)

        assert len(residuals) == len(time_series)
        assert len(std_dev) == len(time_series)
        assert np.all(std_dev > 0)  # Avoid division by zero

    def test_predict_anomaly_scores(self):
        """Test anomaly score computation."""
        detector = FarringtonDetector()
        time_series = np.arange(100) + np.random.poisson(10, 100)

        detector.fit(time_series)
        scores = detector.predict_anomaly_scores(time_series)

        assert len(scores) == len(time_series)
        # Scores should be finite
        assert np.all(np.isfinite(scores))
        # Some scores should be positive (anomalies)
        assert np.sum(scores > 0) > 0

    def test_detect_anomalies(self):
        """Test anomaly detection."""
        detector = FarringtonDetector()
        time_series = np.arange(100) + np.random.poisson(10, 100)

        detector.fit(time_series)
        anomalies = detector.detect_anomalies(time_series)

        assert len(anomalies) == len(time_series)
        assert np.all(anomalies >= 0)  # Binary flags
        assert np.all(anomalies <= 1)  # Binary flags

    def test_score(self):
        """Test score method."""
        detector = FarringtonDetector()
        time_series = np.arange(100) + np.random.poisson(10, 100)

        detector.fit(time_series)
        scores = detector.score(time_series)

        assert len(scores) == len(time_series)
        assert np.all(np.isfinite(scores))  # No inf/nan

    def test_score_with_population(self):
        """Test score with population offset."""
        detector = FarringtonDetector()
        time_series = np.arange(100) + np.random.poisson(10, 100)
        population = np.random.uniform(1000, 10000, 100)

        detector.fit(time_series)
        scores = detector.score(time_series, population)

        assert len(scores) == len(time_series)

    def test_predict_before_fit(self):
        """Test prediction before fit raises error."""
        detector = FarringtonDetector()

        with pytest.raises(ValueError, match="Model must be fitted before prediction"):
            detector.predict_expected(np.arange(50))

        with pytest.raises(ValueError, match="Model must be fitted before prediction"):
            detector.predict_std_residuals(np.arange(50))

        with pytest.raises(ValueError, match="Model must be fitted before prediction"):
            detector.predict_anomaly_scores(np.arange(50))

        with pytest.raises(ValueError, match="Model must be fitted before detection"):
            detector.detect_anomalies(np.arange(50))

        with pytest.raises(ValueError, match="Model must be fitted before prediction"):
            detector.score(np.arange(50))


class TestFarringtonRealData:
    """Test on real epidemiological time series data."""

    def test_synthetic_anomaly_detection(self):
        """Test detection on synthetic data with injected anomalies."""
        # Generate base time series
        np.random.seed(42)
        n = 104  # 52 weeks * 2 years
        time = np.arange(n)
        base_trend = 10 + 0.05 * time
        base_seasonal = 2 * np.sin(2 * np.pi * time / 52)
        expected = base_trend + base_seasonal + 5  # Baseline rate

        # Inject anomalies
        np.random.seed(42)
        observed = np.random.poisson(expected)
        observed[50:52] += 50  # Spike in week 51-52
        observed[75:77] += 60  # Spike in week 76-77

        # Fit detector
        detector = FarringtonDetector()
        detector.fit(observed)

        # Detect anomalies
        anomalies = detector.detect_anomalies(observed)

        # Should detect injected anomalies
        assert np.any(anomalies[50:52] == 1), "Should detect spike in week 51-52"
        assert np.any(anomalies[75:77] == 1), "Should detect spike in week 76-77"

        # Compute anomaly scores
        scores = detector.predict_anomaly_scores(observed)
        assert np.max(scores) > 3.0, "Anomaly scores should exceed threshold"

    def test_zscore_threshold(self):
        """Test that Z-score threshold works correctly."""
        np.random.seed(42)
        n = 52  # One year
        time = np.arange(n)
        base_trend = 10 + 0.02 * time
        expected = base_trend + 5
        observed = np.random.poisson(expected)

        # Fit detector with custom threshold
        config = FarringtonConfig(z_threshold=4.0)
        detector = FarringtonDetector(config=config)
        detector.fit(observed)

        anomalies = detector.detect_anomalies(observed)
        # With more lenient threshold, we might detect fewer anomalies
        assert len(anomalies) == n

    def test_llr_test(self):
        """Test log-likelihood ratio test."""
        np.random.seed(42)
        n = 104
        time = np.arange(n)
        base_trend = 10 + 0.05 * time
        base_seasonal = 2 * np.sin(2 * np.pi * time / 52)
        expected = base_trend + base_seasonal + 5
        observed = np.random.poisson(expected)

        # Fit detector
        detector = FarringtonDetector()
        detector.fit(observed)

        # Detect anomalies using LLR test
        anomalies_llr = detector.detect_anomalies(observed)

        # Should produce valid flags
        assert len(anomalies_llr) == n
        assert np.all(anomalies_llr >= 0)
        assert np.all(anomalies_llr <= 1)


class TestFarringtonRegistry:
    """Test registry integration."""

    def test_registered_detector_available(self):
        """Test that detector is registered."""
        from cdade.registry import list_baseline_detectors

        assert "farrington" in list_baseline_detectors()
        assert len(list_baseline_detectors()) >= 1


def test_baselines_writes_to_namespaced_dir(tmp_path, monkeypatch):
    """After running baselines, scores appear under results/baselines/{dataset}/."""
    from types import SimpleNamespace
    from unittest.mock import MagicMock, patch

    import numpy as np
    import pandas as pd

    from cdade.baselines import run_baselines

    # Build minimal synthetic data
    n = 50
    counts = pd.DataFrame(np.random.default_rng(0).uniform(0, 10, (n, 3)))
    mask = pd.DataFrame(np.zeros((n, 3), dtype=bool))

    cfg = SimpleNamespace(
        datasets=SimpleNamespace(active=["sivep"]),
        experiment=SimpleNamespace(seed=42, mlflow_tracking_uri="sqlite:///test.db"),
        detector=SimpleNamespace(name="iforest"),
    )

    monkeypatch.setattr(run_baselines, "_PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(run_baselines, "_RESULTS_DIR", tmp_path / "results" / "baselines")

    with patch.object(run_baselines, "_load_injected_data", return_value=(counts, mask)), patch(
        "mlflow.set_tracking_uri"
    ), patch("mlflow.set_experiment"), patch(
        "mlflow.start_run",
        return_value=MagicMock(
            __enter__=MagicMock(return_value=MagicMock()), __exit__=MagicMock(return_value=False)
        ),
    ):
        run_baselines._run_baselines_for_dataset(
            "sivep", counts, mask, cfg, tmp_path / "results" / "baselines" / "sivep"
        )

    sivep_dir = tmp_path / "results" / "baselines" / "sivep"
    # At least the directory should exist after running for a dataset
    assert sivep_dir.exists()
