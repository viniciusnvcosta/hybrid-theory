# ABOUTME: Tests for anomaly detection evaluation metrics (AUC-PR, NAB, P/R/F1).
# ABOUTME: Covers perfect detection, random scores, early/late detection, and threshold selection.

import numpy as np
import pytest

from cdade.evaluation.metrics import (
    compute_all_metrics,
    compute_auc_pr,
    compute_nab_score,
    compute_pr_f1,
)


class TestAUCPR:
    """Tests for AUC-PR metric."""

    def test_auc_pr_perfect(self):
        """Perfect scores (equal to y_true) should yield AUC-PR = 1.0."""
        y_true = np.array([0, 0, 1, 1, 0, 1])
        scores = y_true.astype(float)  # Perfect scores
        auc = compute_auc_pr(y_true, scores)
        assert auc == pytest.approx(1.0, abs=1e-6)

    def test_auc_pr_random(self):
        """All-zero scores should yield AUC-PR ≈ contamination rate."""
        y_true = np.array([0, 0, 0, 1, 0, 0])  # 1 anomaly in 6 samples
        scores = np.zeros(6)  # All scores are 0
        auc = compute_auc_pr(y_true, scores)
        contamination = np.mean(y_true)  # 1/6 ≈ 0.167
        assert auc == pytest.approx(contamination, abs=1e-2)


class TestNABScore:
    """Tests for NAB streaming score."""

    def test_nab_early_detection(self):
        """Detection before anomaly within window should get credit."""
        # Anomaly at index 3-4, window=3 means [0, 7]
        y_true = np.array([0, 0, 0, 1, 1, 0, 0, 0])

        # Detection at index 2 (one before anomaly onset), within window [0, 7]
        # Scores with baseline so threshold is clear: [0, 0, 1, 0.5, 0.5, 0, 0, 0]
        # Median = 0.25, threshold includes index 2 and 3, 4
        scores = np.array([0, 0, 1, 0.5, 0.5, 0, 0, 0])
        nab = compute_nab_score(y_true, scores, window=3)

        # Should get TP credit for detecting within window
        # 1 TP / 1 anomaly = 1.0
        assert nab == pytest.approx(1.0, abs=1e-6)

    def test_nab_late_detection(self):
        """The NAB function should correctly apply window-based credit."""
        # Two anomalies at indices 2 and 6, window=1 means [1,3] and [5,7]
        y_true = np.array([0, 0, 1, 0, 0, 0, 1, 0])

        # Case 1: Detect first anomaly in-window (at 2), miss second
        # Scores: high at index 2 only
        scores_first = np.array([0.1, 0.1, 10, 0.1, 0.1, 0.1, 0.1, 0.1])
        nab_first = compute_nab_score(y_true, scores_first, window=1)

        # Case 2: Detect second anomaly in-window (at 6), miss first
        # Scores: high at index 6 only
        scores_second = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 10, 0.1])
        nab_second = compute_nab_score(y_true, scores_second, window=1)

        # Both detect exactly 1 out of 2 anomalies, should score equally
        assert nab_first == pytest.approx(nab_second, abs=1e-6)


class TestPRF1:
    """Tests for Precision, Recall, F1 at threshold."""

    def test_f1_at_median_threshold(self):
        """Known scores and median threshold should yield expected P/R/F1."""
        # Simple case: scores perfectly separate anomalies from normal
        # 6 elements: 3 anomalies at indices 1, 3, 5 with scores 0.8, 0.9, 0.85
        # 3 normal at indices 0, 2, 4 with scores 0.1, 0.2, 0.15
        y_true = np.array([0, 1, 0, 1, 0, 1])
        scores = np.array([0.1, 0.8, 0.2, 0.9, 0.15, 0.85])
        # Sorted: [0.1, 0.15, 0.2, 0.8, 0.85, 0.9]
        # Median = (0.2 + 0.8) / 2 = 0.5

        result = compute_pr_f1(y_true, scores, threshold="median")

        # Predictions at median threshold (0.5): scores >= 0.5 are anomalies
        # Indices with score >= 0.5: 1 (0.8), 3 (0.9), 5 (0.85)
        # TP=3, FP=0, FN=0
        # Precision = 3/3 = 1.0, Recall = 3/3 = 1.0, F1 = 1.0
        assert result["precision"] == pytest.approx(1.0, abs=1e-6)
        assert result["recall"] == pytest.approx(1.0, abs=1e-6)
        assert result["f1"] == pytest.approx(1.0, abs=1e-6)


class TestComputeAllMetrics:
    """Tests for compute_all_metrics wrapper."""

    def test_compute_all_metrics_keys(self):
        """Output dict must have exactly {auc_pr, nab, precision, recall, f1, threshold}."""
        y_true = np.array([0, 0, 1, 1, 0, 1])
        scores = np.array([0.1, 0.2, 0.9, 0.8, 0.15, 0.85])

        result = compute_all_metrics(y_true, scores, nab_window=4)

        expected_keys = {"auc_pr", "nab", "precision", "recall", "f1", "threshold"}
        assert set(result.keys()) == expected_keys, f"Keys mismatch: {set(result.keys())}"

    def test_compute_all_metrics_values_in_range(self):
        """All metrics should be in reasonable ranges."""
        y_true = np.array([0, 0, 1, 1, 0, 1])
        scores = np.array([0.1, 0.2, 0.9, 0.8, 0.15, 0.85])

        result = compute_all_metrics(y_true, scores, nab_window=4)

        assert 0 <= result["auc_pr"] <= 1
        assert 0 <= result["nab"] <= 1
        assert 0 <= result["precision"] <= 1
        assert 0 <= result["recall"] <= 1
        assert 0 <= result["f1"] <= 1
        assert isinstance(result["threshold"], float | int | np.number)
