# ABOUTME: Tests for the run_evaluate module that orchestrates evaluation across methods.
# ABOUTME: Uses tmp_path fixtures with synthetic data; does not depend on real files.

import json

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def tmp_project_root(tmp_path):
    """Create a minimal temp project structure."""
    (tmp_path / "data" / "injected").mkdir(parents=True)
    (tmp_path / "results" / "baselines").mkdir(parents=True)
    (tmp_path / "results" / "selection").mkdir(parents=True)
    return tmp_path


@pytest.fixture
def synthetic_mask_data(tmp_project_root):
    """Synthetic ground truth mask: 132 time steps, 13 series."""
    mask = np.zeros((132, 13), dtype=bool)
    # Inject some anomalies in last 26 indices (test set)
    mask[106:112, 0] = True  # Spike in series 0
    mask[120:125, 5] = True  # Spike in series 5

    mask_df = pd.DataFrame(mask)
    mask_path = tmp_project_root / "data" / "injected" / "sivep_counts_mask.parquet"
    mask_df.to_parquet(mask_path)
    return mask_path


@pytest.fixture
def synthetic_blended_scores(tmp_project_root):
    """Synthetic blended scores: 132 time steps, 13 detectors."""
    rng = np.random.default_rng(42)
    scores = rng.uniform(0, 1, (132, 13))
    scores[106:112, :] += 0.3  # Inject signal in test region

    df = pd.DataFrame(scores, columns=[f"detector_{i}" for i in range(13)])
    blended_path = tmp_project_root / "results" / "selection" / "blended_scores.csv"
    df.to_csv(blended_path, index=False)
    return blended_path


@pytest.fixture
def synthetic_baseline_scores(tmp_project_root):
    """Synthetic baseline scores b1–b5: each is a 26-element npy array."""
    rng = np.random.default_rng(42)
    baseline_dir = tmp_project_root / "results" / "baselines"
    for i in range(1, 6):
        scores = rng.uniform(0, 1, 26)
        scores[0:5] += 0.2  # Inject some signal
        np.save(baseline_dir / f"b{i}_scores.npy", scores)
    return baseline_dir


class TestLoadDataRaisesIfMissing:
    """Test that missing injected data raises FileNotFoundError with clear message."""

    def test_missing_mask_raises(self, tmp_project_root):
        """When mask file is absent, load should raise FileNotFoundError."""
        from cdade.evaluation.run_evaluate import load_ground_truth

        with pytest.raises(FileNotFoundError, match="sivep_counts_mask.parquet"):
            load_ground_truth(tmp_project_root / "data" / "injected")


class TestCDADEScoresAlignment:
    """Test that blended_scores shape mismatch raises ValueError with helpful message."""

    def test_blended_scores_wrong_rows(self, tmp_project_root, synthetic_mask_data):
        """When blended_scores has wrong number of rows, should raise ValueError."""
        from cdade.evaluation.run_evaluate import load_blended_scores, load_ground_truth

        y_true = load_ground_truth(tmp_project_root / "data" / "injected")

        # Create a blended_scores with wrong number of rows
        bad_scores = np.random.uniform(0, 1, (100, 13))  # 100 instead of 132
        bad_df = pd.DataFrame(bad_scores, columns=[f"detector_{i}" for i in range(13)])
        blended_path = tmp_project_root / "results" / "selection" / "blended_scores.csv"
        bad_df.to_csv(blended_path, index=False)

        with pytest.raises(ValueError, match="shape mismatch|length"):
            load_blended_scores(blended_path, len(y_true))


class TestMetricsJSONStructure:
    """Test that metrics.json has correct structure with all methods."""

    def test_metrics_json_structure(
        self,
        tmp_project_root,
        synthetic_mask_data,
        synthetic_blended_scores,
        synthetic_baseline_scores,
    ):
        """After evaluation, metrics.json should have keys for cdade + b1–b5."""
        from cdade.evaluation.run_evaluate import (
            evaluate_all_methods,
            load_baseline_scores,
            load_blended_scores,
            load_ground_truth,
        )

        y_true = load_ground_truth(tmp_project_root / "data" / "injected")
        blended_df = load_blended_scores(synthetic_blended_scores, len(y_true))
        baselines = load_baseline_scores(tmp_project_root / "results" / "baselines")

        # Test/train split: last 26 are test
        y_test = y_true[106:]
        cdade_test = blended_df.values[106:].max(axis=1)

        nab_window = 4
        all_metrics = evaluate_all_methods(y_test, cdade_test, baselines, nab_window)

        # Should have keys for cdade + b1–b5
        expected_keys = {"cdade", "b1", "b2", "b3", "b4", "b5"}
        assert set(all_metrics.keys()) == expected_keys, f"Keys mismatch: {set(all_metrics.keys())}"

        # Each value should be a dict with metric keys
        for method, metrics in all_metrics.items():
            assert isinstance(metrics, dict), f"{method} should have dict metrics"
            assert "auc_pr" in metrics, f"{method} missing auc_pr"
            assert "nab" in metrics, f"{method} missing nab"


class TestEvaluationOutputDirCreated:
    """Test that results/evaluation/ is created with per-method csvs."""

    def test_evaluation_output_dir_created(
        self,
        tmp_project_root,
        synthetic_mask_data,
        synthetic_blended_scores,
        synthetic_baseline_scores,
    ):
        """After evaluation, results/evaluation/ should exist with per-method csvs."""
        from cdade.evaluation.run_evaluate import (
            evaluate_all_methods,
            load_baseline_scores,
            load_blended_scores,
            load_ground_truth,
            save_per_method_csvs,
        )

        # Load data
        y_true = load_ground_truth(tmp_project_root / "data" / "injected")
        blended_df = load_blended_scores(synthetic_blended_scores, len(y_true))
        baselines = load_baseline_scores(tmp_project_root / "results" / "baselines")

        # Test/train split
        y_test = y_true[106:]
        cdade_test = blended_df.values[106:].max(axis=1)

        # Evaluate
        nab_window = 4
        all_metrics = evaluate_all_methods(y_test, cdade_test, baselines, nab_window)

        # Save CSVs
        eval_dir = tmp_project_root / "results" / "evaluation"
        save_per_method_csvs(all_metrics, eval_dir)

        # Verify dir exists and has per-method csvs
        assert eval_dir.exists(), f"Evaluation dir not created at {eval_dir}"

        for method in ["cdade", "b1", "b2", "b3", "b4", "b5"]:
            csv_path = eval_dir / f"{method}_metrics.csv"
            assert csv_path.exists(), f"Missing {method}_metrics.csv"

            # Verify content is parseable
            df = pd.read_csv(csv_path)
            assert len(df) > 0, f"{method}_metrics.csv is empty"


class TestMetricsJSONOutput:
    """Test that metrics.json is written correctly."""

    def test_save_metrics_json(
        self,
        tmp_project_root,
        synthetic_mask_data,
        synthetic_blended_scores,
        synthetic_baseline_scores,
    ):
        """metrics.json should be written with all methods and be parseable."""
        from cdade.evaluation.run_evaluate import (
            evaluate_all_methods,
            load_baseline_scores,
            load_blended_scores,
            load_ground_truth,
            save_metrics_json,
        )

        # Load and evaluate
        y_true = load_ground_truth(tmp_project_root / "data" / "injected")
        blended_df = load_blended_scores(synthetic_blended_scores, len(y_true))
        baselines = load_baseline_scores(tmp_project_root / "results" / "baselines")

        y_test = y_true[106:]
        cdade_test = blended_df.values[106:].max(axis=1)

        nab_window = 4
        all_metrics = evaluate_all_methods(y_test, cdade_test, baselines, nab_window)

        # Save JSON
        metrics_path = tmp_project_root / "results" / "metrics.json"
        save_metrics_json(all_metrics, metrics_path)

        # Verify file exists and is valid JSON
        assert metrics_path.exists(), "metrics.json not created"

        with open(metrics_path) as fh:
            loaded = json.load(fh)

        assert set(loaded.keys()) == {"cdade", "b1", "b2", "b3", "b4", "b5"}
        for _method, metrics in loaded.items():
            assert isinstance(metrics, dict)
            assert "auc_pr" in metrics
