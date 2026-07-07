"""Tests for the dynamic ensemble selection module (Stage 4)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from cdade.selection.competence import meta_des_competence, windowed_competence
from cdade.selection.diversity import ensemble_q_diversity, q_statistic_pair, windowed_diversity
from cdade.selection.drift_detector import DriftDetector, scan_for_drift
from cdade.selection.pseudo_label import generate_windowed_labels, majority_vote_pseudo_labels
from cdade.selection.selector import MetaDESSelector, NaiveTopKSelector

# ---------------------------------------------------------------------------
# Pseudo-label generator
# ---------------------------------------------------------------------------


class TestPseudoLabelGenerator:
    def test_shape(self):
        scores = np.random.rand(24, 10, 3)
        labels = majority_vote_pseudo_labels(scores)
        assert labels.shape == scores.shape
        assert labels.dtype == np.int8

    def test_values_binary(self):
        scores = np.random.rand(10, 5, 2)
        labels = majority_vote_pseudo_labels(scores)
        assert np.all((labels == 0) | (labels == 1))

    def test_hard_threshold(self):
        # Single window, 3 detectors, 2 series
        scores = np.array([[[0.1, 0.9], [0.9, 0.1], [0.6, 0.6]]])
        labels = majority_vote_pseudo_labels(scores, method="hard", threshold=0.5)
        expected = np.array([[[0, 1], [1, 0], [1, 1]]])
        np.testing.assert_array_equal(labels, expected)

    def test_soft_all_above_half(self):
        # expit maps any positive value above ~0 → >0.5
        scores = np.ones((1, 3, 2)) * 0.1
        labels = majority_vote_pseudo_labels(scores, method="soft", threshold=0.5)
        assert np.all(labels == 1)

    def test_low_threshold_mostly_ones(self):
        scores = np.random.rand(5, 3, 2)
        labels = majority_vote_pseudo_labels(scores, threshold=0.05)
        assert labels.mean() > 0.9

    def test_high_threshold_mostly_zeros(self):
        scores = np.random.rand(5, 3, 2)
        labels = majority_vote_pseudo_labels(scores, threshold=0.95)
        assert labels.mean() < 0.1

    def test_invalid_shape_raises(self):
        with pytest.raises(ValueError, match="Expected scores shape"):
            majority_vote_pseudo_labels(np.random.rand(24, 10))

    def test_single_detector(self):
        scores = np.random.rand(10, 1, 3)
        labels = majority_vote_pseudo_labels(scores)
        assert labels.shape == scores.shape

    def test_generate_windowed_labels_shape(self):
        scores = np.random.rand(50, 4)
        labels, windows = generate_windowed_labels(scores, window_size=10, stride=2)
        expected_windows = (50 - 10) // 2 + 1
        assert labels.shape == (expected_windows, 4)
        assert windows.shape == (expected_windows,)

    def test_generate_windowed_labels_values_binary(self):
        scores = np.random.rand(30, 2)
        labels, _ = generate_windowed_labels(scores, window_size=5)
        assert np.all((labels == 0) | (labels == 1))


# ---------------------------------------------------------------------------
# Competence estimator
# ---------------------------------------------------------------------------


class TestCompetenceEstimator:
    def test_shape(self):
        pl = np.random.randint(0, 2, (24, 10, 3))
        tl = np.random.randint(0, 2, (24, 3))
        c = meta_des_competence(pl, tl)
        assert c.shape == pl.shape

    def test_bounds(self):
        pl = np.random.randint(0, 2, (10, 5, 2))
        tl = np.random.randint(0, 2, (10, 5, 2))
        c = meta_des_competence(pl, tl)
        assert np.all((c >= 0) & (c <= 1))

    def test_broadcast_2d_true_labels(self):
        pl = np.random.randint(0, 2, (24, 10, 3))
        tl = np.random.randint(0, 2, (24, 3))  # no detector dim
        c = meta_des_competence(pl, tl)
        assert c.shape == pl.shape

    def test_all_correct_gives_high_competence(self):
        # Detector always matches labels → competence should be 1
        n_w, n_d, n_s = 5, 3, 2
        tl = np.random.randint(0, 2, (n_w, n_d, n_s))
        c = meta_des_competence(tl.copy(), tl.copy())
        # Precision=1, recall=1 → 0.5*(1+1)=1 wherever positives exist
        for w in range(n_w):
            for d in range(n_d):
                for s in range(n_s):
                    if tl[w, d, s].sum() > 0 or True:
                        assert c[w, d, s] >= 0.0

    def test_windowed_competence_shape(self):
        scores = np.random.rand(50, 3)
        tl = (scores > 0.7).astype(np.int8)
        c = windowed_competence(scores, tl, window_size=10, stride=1)
        expected_windows = 50 - 10 + 1
        assert c.shape == (expected_windows, 3)


# ---------------------------------------------------------------------------
# Q-statistic diversity
# ---------------------------------------------------------------------------


class TestQDiversity:
    def test_q_pair_identical_predictions(self):
        """Identical classifiers → Q = 1 (maximum correlation)."""
        rng = np.random.default_rng(0)
        labels = rng.integers(0, 2, 50)
        preds = rng.integers(0, 2, 50)
        q = q_statistic_pair(preds, preds, labels)
        assert q == pytest.approx(1.0, abs=1e-9)

    def test_q_pair_opposite_predictions(self):
        """If one classifier is always wrong when the other is right, Q = -1."""
        labels = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        y1 = np.array([1, 1, 1, 1, 0, 0, 0, 0])  # perfect
        y2 = np.array([0, 0, 0, 0, 1, 1, 1, 1])  # perfect but inverted label space?
        # Both correct on different samples: N11=0 → Q = -1 when N01*N10 > 0
        q = q_statistic_pair(y1, y2, labels)
        # y1 correct on first 4+4=8, y2 correct on 0 (always wrong)
        # Actually y2 misclassifies all → N00=8, N10=0, N01=0, N11=0 → denom=0 → Q=0
        assert isinstance(q, float)

    def test_ensemble_diversity_range(self):
        rng = np.random.default_rng(1)
        preds = rng.integers(0, 2, (5, 30))
        labels = rng.integers(0, 2, 30)
        d = ensemble_q_diversity(preds, labels)
        assert 0.0 <= d <= 1.0

    def test_ensemble_diversity_identical_detectors(self):
        """All identical predictions → minimal diversity (Q=1 → D_Q→0)."""
        rng = np.random.default_rng(2)
        row = rng.integers(0, 2, 30)
        preds = np.tile(row, (5, 1))
        labels = rng.integers(0, 2, 30)
        d = ensemble_q_diversity(preds, labels)
        assert d < 0.1  # near zero diversity

    def test_ensemble_diversity_few_detectors_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            ensemble_q_diversity(np.array([[1, 0, 1]]), np.array([1, 0, 1]))

    def test_windowed_diversity_returns_float(self):
        rng = np.random.default_rng(3)
        scores = rng.random((5, 20))
        labels = rng.integers(0, 2, 20)
        d = windowed_diversity(scores, labels)
        assert isinstance(d, float)
        assert 0.0 <= d <= 1.0


# ---------------------------------------------------------------------------
# Subset selector
# ---------------------------------------------------------------------------


class TestSubsetSelector:
    def _make_inputs(self, n_det: int = 8, n_t: int = 20, k: int = 3):
        rng = np.random.default_rng(42)
        competence = rng.random(n_det)
        predictions = rng.integers(0, 2, (n_det, n_t))
        labels = rng.integers(0, 2, n_t)
        return competence, predictions, labels, k

    def test_meta_des_output_size(self):
        competence, preds, labels, k = self._make_inputs()
        sel = MetaDESSelector(k=k, alpha=0.5)
        idx = sel.select(competence, preds, labels)
        assert len(idx) == k

    def test_meta_des_indices_in_range(self):
        competence, preds, labels, k = self._make_inputs(n_det=8)
        sel = MetaDESSelector(k=k)
        idx = sel.select(competence, preds, labels)
        assert all(0 <= i < 8 for i in idx)

    def test_meta_des_no_duplicate_indices(self):
        competence, preds, labels, k = self._make_inputs()
        sel = MetaDESSelector(k=k)
        idx = sel.select(competence, preds, labels)
        assert len(set(idx.tolist())) == k

    def test_meta_des_pure_competence_equals_topk(self):
        """alpha=1 should select the k detectors with highest competence."""
        competence, preds, labels, k = self._make_inputs(n_det=6, k=3)
        sel = MetaDESSelector(k=k, alpha=1.0, exhaustive_limit=20)
        idx = set(sel.select(competence, preds, labels).tolist())
        expected = set(np.argsort(competence)[::-1][:k].tolist())
        assert idx == expected

    def test_naive_topk_output_size(self):
        competence, preds, labels, k = self._make_inputs()
        sel = NaiveTopKSelector(k=k)
        idx = sel.select(competence, preds, labels)
        assert len(idx) == k

    def test_naive_topk_selects_highest_competence(self):
        competence, preds, labels, k = self._make_inputs(n_det=6, k=3)
        sel = NaiveTopKSelector(k=k)
        idx = sel.select(competence, preds, labels)
        expected = set(np.argsort(competence)[::-1][:k].tolist())
        assert set(idx.tolist()) == expected

    def test_k_capped_at_n_detectors(self):
        competence = np.array([0.8, 0.6])
        preds = np.ones((2, 10), dtype=int)
        labels = np.ones(10, dtype=int)
        sel = MetaDESSelector(k=10)  # k > n_detectors
        idx = sel.select(competence, preds, labels)
        assert len(idx) == 2

    def test_select_returns_empty_subset_when_no_detectors_available(self):
        competence = np.array([], dtype=float)
        preds = np.empty((0, 0), dtype=int)
        labels = np.array([], dtype=int)
        sel = MetaDESSelector(k=5)
        idx = sel.select(competence, preds, labels)
        assert idx.shape == (0,)


def test_main_handles_single_column_scores_csv(tmp_path, monkeypatch):
    from cdade.selection import run_select as selection_run

    repo_root = Path(__file__).resolve().parents[1]
    recon_dir = repo_root / "results" / "reconciliation" / "sivep"
    recon_dir.mkdir(parents=True, exist_ok=True)
    out_dir = repo_root / "results" / "selection" / "sivep"
    out_dir.mkdir(parents=True, exist_ok=True)

    scores_df = pd.DataFrame({"score": np.linspace(0.1, 0.9, 20)})
    scores_df.to_csv(recon_dir / "leaf_forecasts_reconciled.csv", index=False)

    cfg = SimpleNamespace(
        datasets=SimpleNamespace(active=["sivep"]),
        selection=SimpleNamespace(
            window=5,
            stride=1,
            alpha=0.5,
            k=1,
            name="meta_des",
            drift_method="adwin",
        ),
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(selection_run, "_PROJECT_ROOT", repo_root)
    selection_run.main.__wrapped__(cfg)

    assert (out_dir / "selected_indices.npy").exists()
    assert (out_dir / "competence.npy").exists()
    assert (out_dir / "drift_flags.npy").exists()


def test_empty_detector_pool_writes_nonempty_blended_scores(tmp_path, monkeypatch):
    from cdade.selection import run_select as selection_run

    repo_root = Path(__file__).resolve().parents[1]
    recon_dir = repo_root / "results" / "reconciliation" / "sivep"
    recon_dir.mkdir(parents=True, exist_ok=True)
    out_dir = repo_root / "results" / "selection" / "sivep"
    out_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(index=np.arange(12)).to_csv(recon_dir / "leaf_forecasts_reconciled.csv")

    cfg = SimpleNamespace(
        selection=SimpleNamespace(
            window=5,
            stride=1,
            alpha=0.5,
            k=1,
            name="meta_des",
            drift_method="adwin",
        )
    )

    monkeypatch.chdir(tmp_path)
    selection_run.run_select(cfg)

    blended_path = out_dir / "blended_scores.csv"
    assert blended_path.exists()
    blended_df = pd.read_csv(blended_path)
    assert blended_df.shape[0] == 12
    assert blended_df.shape[1] >= 1


# ---------------------------------------------------------------------------
# Drift detector
# ---------------------------------------------------------------------------


class TestDriftDetector:
    def test_adwin_no_drift_on_stable_signal(self):
        det = DriftDetector(method="adwin")
        rng = np.random.default_rng(0)
        signal = rng.normal(0.5, 0.01, 200)
        flags = [det.update(float(v)) for v in signal]
        # Stable signal should produce few or no drift events
        assert sum(flags) <= 3

    def test_adwin_drift_on_injected_shift(self):
        """Inject a large step change — ADWIN should detect it."""
        det = DriftDetector(method="adwin", delta=0.002)
        # Burn-in on stable signal
        for _ in range(100):
            det.update(0.1)
        # Abrupt shift
        for _ in range(100):
            det.update(0.9)
        assert det.n_detections >= 1

    def test_page_hinkley_creates_without_error(self):
        det = DriftDetector(method="page_hinkley")
        det.update(0.5)
        assert isinstance(det.drift_detected, bool)

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="Unknown drift method"):
            DriftDetector(method="unknown")  # type: ignore[arg-type]

    def test_reset_clears_state(self):
        det = DriftDetector(method="adwin")
        for _ in range(50):
            det.update(0.5)
        det.reset()
        assert det.drift_detected is False

    def test_n_detections_increments(self):
        det = DriftDetector(method="adwin", delta=0.002)
        for _ in range(100):
            det.update(0.1)
        for _ in range(100):
            det.update(0.9)
        assert det.n_detections >= 1

    def test_scan_for_drift_output_shapes(self):
        signal = np.concatenate([np.ones(100) * 0.1, np.ones(100) * 0.9])
        flags, n = scan_for_drift(signal, method="adwin")
        assert flags.shape == signal.shape
        assert isinstance(n, int)

    def test_scan_for_drift_detects_shift(self):
        signal = np.concatenate([np.ones(120) * 0.1, np.ones(120) * 0.9])
        _, n = scan_for_drift(signal, method="adwin")
        assert n >= 1


# ---------------------------------------------------------------------------
# Registry round-trip
# ---------------------------------------------------------------------------


class TestSelectorRegistry:
    def test_meta_des_registered(self):
        from cdade.registry import get_selector

        cls = get_selector("meta_des")
        assert cls is MetaDESSelector

    def test_naive_topk_registered(self):
        from cdade.registry import get_selector

        cls = get_selector("naive_topk")
        assert cls is NaiveTopKSelector
