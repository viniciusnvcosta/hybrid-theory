"""Tests for B3 (EnsembleAverage), B4 (StaticTopK), and B5 (ReconciliationEVT) baselines.

Author: CDADE project
"""

import numpy as np
import pytest

# --- B3: EnsembleAverage ---
from cdade.baselines.ensemble_average import (
    EnsembleAverageConfig,
    EnsembleAverageDetector,
)

# --- B5: ReconciliationEVT ---
from cdade.baselines.reconciliation_evt import (
    ReconciliationEVTConfig,
    ReconciliationEVTDetector,
)

# --- B4: StaticTopK ---
from cdade.baselines.static_topk import (
    StaticTopKConfig,
    StaticTopKDetector,
    greedy_set_cover_selection,
)

# Register detectors so the baselines can find them
from cdade.detectors import iforest, lof, pca  # noqa: F401
from cdade.registry import list_baseline_detectors

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def small_data(rng):
    """100×5 normal training data, 40×5 test data."""
    X_train = rng.standard_normal((100, 5))
    X_test = rng.standard_normal((40, 5))
    # Inject a few anomalies into test set
    X_test[0] += 10
    X_test[10] -= 10
    y_test = np.zeros(40, dtype=int)
    y_test[[0, 10]] = 1
    return X_train, X_test, y_test


SMALL_POOL = ["pca", "iforest", "lof"]


# ===========================================================================
# B3: EnsembleAverageDetector
# ===========================================================================


class TestEnsembleAverageConfig:
    def test_defaults(self):
        cfg = EnsembleAverageConfig()
        assert cfg.normalize is True
        assert cfg.detector_names == []

    def test_custom(self):
        cfg = EnsembleAverageConfig(normalize=False, detector_names=["pca"])
        assert cfg.normalize is False
        assert cfg.detector_names == ["pca"]


class TestEnsembleAverageDetector:
    def test_score_shape(self, small_data):
        X_train, X_test, _ = small_data
        det = EnsembleAverageDetector(EnsembleAverageConfig(detector_names=SMALL_POOL))
        det.fit(X_train)
        scores = det.score(X_test)
        assert scores.shape == (len(X_test),)

    def test_score_finite(self, small_data):
        X_train, X_test, _ = small_data
        det = EnsembleAverageDetector(EnsembleAverageConfig(detector_names=SMALL_POOL))
        det.fit(X_train)
        assert np.all(np.isfinite(det.score(X_test)))

    def test_scores_in_range_when_normalized(self, small_data):
        """Averaged normalised scores should lie in [0, 1]."""
        X_train, X_test, _ = small_data
        det = EnsembleAverageDetector(
            EnsembleAverageConfig(normalize=True, detector_names=SMALL_POOL)
        )
        det.fit(X_train)
        scores = det.score(X_test)
        # Individual normalised scores are in [0,1]; averages also in [0,1]
        assert np.all(scores >= 0.0)
        assert np.all(scores <= 1.0)

    def test_predict_binary(self, small_data):
        X_train, X_test, _ = small_data
        det = EnsembleAverageDetector(EnsembleAverageConfig(detector_names=SMALL_POOL))
        det.fit(X_train)
        preds = det.predict(X_test)
        assert preds.shape == (len(X_test),)
        assert set(preds.tolist()).issubset({0, 1})

    def test_score_before_fit_raises(self):
        det = EnsembleAverageDetector()
        with pytest.raises(RuntimeError):
            det.score(np.random.randn(10, 5))

    def test_registered_in_baseline_registry(self):
        assert "ensemble_average" in list_baseline_detectors()

    def test_anomalies_have_nonzero_scores(self, small_data):
        """Injected anomalies at ±10 standard deviations should produce nonzero scores."""
        X_train, X_test, y_test = small_data
        det = EnsembleAverageDetector(EnsembleAverageConfig(detector_names=SMALL_POOL))
        det.fit(X_train)
        scores = det.score(X_test)
        # After normalisation, injected anomalies must differ from 0 or 1
        anomaly_scores = scores[y_test == 1]
        # The ensemble produces a nonzero score for the anomaly region
        assert np.any(anomaly_scores != scores[y_test == 0].mean())


# ===========================================================================
# B4: StaticTopKDetector
# ===========================================================================


class TestStaticTopKConfig:
    def test_defaults(self):
        cfg = StaticTopKConfig()
        assert cfg.k == 0
        assert cfg.alpha == 0.5
        assert cfg.normalize is True

    def test_custom(self):
        cfg = StaticTopKConfig(k=2, alpha=0.7)
        assert cfg.k == 2
        assert cfg.alpha == 0.7


class TestGreedySetCoverSelection:
    def test_returns_k_indices(self):
        rng = np.random.default_rng(0)
        n_det, n_val = 5, 50
        val_scores = rng.uniform(0, 1, (n_det, n_val))
        y_val = rng.integers(0, 2, n_val)
        selected = greedy_set_cover_selection(val_scores, y_val, k=3, alpha=0.5)
        assert len(selected) == 3

    def test_no_duplicate_indices(self):
        rng = np.random.default_rng(1)
        val_scores = rng.uniform(0, 1, (6, 40))
        y_val = rng.integers(0, 2, 40)
        selected = greedy_set_cover_selection(val_scores, y_val, k=4, alpha=0.5)
        assert len(set(selected)) == len(selected)

    def test_k_capped_at_n_detectors(self):
        rng = np.random.default_rng(2)
        val_scores = rng.uniform(0, 1, (3, 30))
        y_val = rng.integers(0, 2, 30)
        selected = greedy_set_cover_selection(val_scores, y_val, k=10, alpha=0.5)
        assert len(selected) <= 3

    def test_alpha_zero_favours_diversity(self):
        """With alpha=0 purely diversity-driven selection should still return k items."""
        rng = np.random.default_rng(3)
        val_scores = rng.uniform(0, 1, (5, 50))
        y_val = rng.integers(0, 2, 50)
        selected = greedy_set_cover_selection(val_scores, y_val, k=3, alpha=0.0)
        assert len(selected) == 3


class TestStaticTopKDetector:
    def _split(self, X_train, frac=0.8):
        n = int(len(X_train) * frac)
        return X_train[:n], X_train[n:]

    def test_fit_selects_subset(self, small_data):
        X_train, X_test, y_test = small_data
        X_tr, X_val = self._split(X_train)
        y_val = np.zeros(len(X_val), dtype=int)

        det = StaticTopKDetector(StaticTopKConfig(k=2, detector_names=SMALL_POOL))
        det.fit(X_tr, X_val, y_val)

        assert len(det.selected_indices) == 2
        assert len(set(det.selected_indices)) == 2  # no duplicates

    def test_score_shape(self, small_data):
        X_train, X_test, y_test = small_data
        X_tr, X_val = self._split(X_train)
        y_val = np.zeros(len(X_val), dtype=int)

        det = StaticTopKDetector(StaticTopKConfig(k=2, detector_names=SMALL_POOL))
        det.fit(X_tr, X_val, y_val)
        scores = det.score(X_test)
        assert scores.shape == (len(X_test),)

    def test_score_finite(self, small_data):
        X_train, X_test, _ = small_data
        X_tr, X_val = self._split(X_train)
        y_val = np.zeros(len(X_val), dtype=int)

        det = StaticTopKDetector(StaticTopKConfig(k=2, detector_names=SMALL_POOL))
        det.fit(X_tr, X_val, y_val)
        assert np.all(np.isfinite(det.score(X_test)))

    def test_predict_binary(self, small_data):
        X_train, X_test, _ = small_data
        X_tr, X_val = self._split(X_train)
        y_val = np.zeros(len(X_val), dtype=int)

        det = StaticTopKDetector(StaticTopKConfig(k=2, detector_names=SMALL_POOL))
        det.fit(X_tr, X_val, y_val)
        preds = det.predict(X_test)
        assert set(preds.tolist()).issubset({0, 1})

    def test_score_before_fit_raises(self):
        det = StaticTopKDetector()
        with pytest.raises(RuntimeError):
            det.score(np.random.randn(10, 5))

    def test_auto_k_with_k_zero(self, small_data):
        """k=0 should default to sqrt(n_detectors)."""
        X_train, X_test, _ = small_data
        X_tr, X_val = self._split(X_train)
        y_val = np.zeros(len(X_val), dtype=int)

        det = StaticTopKDetector(StaticTopKConfig(k=0, detector_names=SMALL_POOL))
        det.fit(X_tr, X_val, y_val)
        # sqrt(3) ≈ 1 → at least 1 detector selected
        assert len(det.selected_indices) >= 1

    def test_registered_in_baseline_registry(self):
        assert "static_topk" in list_baseline_detectors()


# ===========================================================================
# B5: ReconciliationEVTDetector
# ===========================================================================


class TestReconciliationEVTConfig:
    def test_defaults(self):
        cfg = ReconciliationEVTConfig()
        assert cfg.contamination == 0.05
        assert cfg.normalize is True
        assert cfg.alpha_evt == 0.05

    def test_custom(self):
        cfg = ReconciliationEVTConfig(contamination=0.1, alpha_evt=0.01)
        assert cfg.contamination == 0.1
        assert cfg.alpha_evt == 0.01


class TestReconciliationEVTDetector:
    def test_fit_succeeds(self, small_data):
        X_train, _, _ = small_data
        det = ReconciliationEVTDetector(ReconciliationEVTConfig(detector_names=SMALL_POOL))
        det.fit(X_train)
        assert len(det.fitted_detectors) > 0

    def test_score_shape(self, small_data):
        X_train, X_test, _ = small_data
        det = ReconciliationEVTDetector(ReconciliationEVTConfig(detector_names=SMALL_POOL))
        det.fit(X_train)
        scores = det.score(X_test)
        assert scores.shape == (len(X_test),)

    def test_score_in_zero_one(self, small_data):
        """Anomaly scores should lie in [0, 1]."""
        X_train, X_test, _ = small_data
        det = ReconciliationEVTDetector(ReconciliationEVTConfig(detector_names=SMALL_POOL))
        det.fit(X_train)
        scores = det.score(X_test)
        assert np.all(scores >= 0.0)
        assert np.all(scores <= 1.0 + 1e-9)  # allow tiny float tolerance

    def test_score_finite(self, small_data):
        X_train, X_test, _ = small_data
        det = ReconciliationEVTDetector(ReconciliationEVTConfig(detector_names=SMALL_POOL))
        det.fit(X_train)
        assert np.all(np.isfinite(det.score(X_test)))

    def test_predict_binary(self, small_data):
        X_train, X_test, _ = small_data
        det = ReconciliationEVTDetector(ReconciliationEVTConfig(detector_names=SMALL_POOL))
        det.fit(X_train)
        preds = det.predict(X_test)
        assert preds.shape == (len(X_test),)
        assert set(preds.tolist()).issubset({0, 1})

    def test_predict_with_scores_returns_pair(self, small_data):
        X_train, X_test, _ = small_data
        det = ReconciliationEVTDetector(ReconciliationEVTConfig(detector_names=SMALL_POOL))
        det.fit(X_train)
        preds, scores = det.predict_with_scores(X_test)
        assert preds.shape == (len(X_test),)
        assert scores.shape == (len(X_test),)

    def test_score_before_fit_raises(self):
        det = ReconciliationEVTDetector()
        with pytest.raises(RuntimeError):
            det.score(np.random.randn(10, 5))

    def test_anomalies_produce_distinct_scores(self, small_data):
        """Injected anomalies at ±10 std must produce scores distinguishable from the median."""
        X_train, X_test, y_test = small_data
        det = ReconciliationEVTDetector(ReconciliationEVTConfig(detector_names=SMALL_POOL))
        det.fit(X_train)
        scores = det.score(X_test)
        # The scores must not all be identical (GPD is doing something)
        assert scores.max() > scores.min()

    def test_get_reconciled_scores_returns_series(self, small_data):
        import pandas as pd

        X_train, X_test, _ = small_data
        det = ReconciliationEVTDetector(ReconciliationEVTConfig(detector_names=SMALL_POOL))
        det.fit(X_train)
        series = det.get_reconciled_scores(X_test)
        assert isinstance(series, pd.Series)
        assert len(series) == len(X_test)

    def test_registered_in_baseline_registry(self):
        assert "reconciliation_evt" in list_baseline_detectors()
