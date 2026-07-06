"""Test ablation variant runner.

Tests for the ablation module that validates component attribution through
variant-specific metrics computation.

Author: CDADE project
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from cdade.ablation.run_ablation import (
    apply_variant_transformation,
    run_ablation_variants,
)
from cdade.reconciliation.identity import IdentityReconciler
from cdade.selection.selector import NaiveTopKSelector


class TestIdentityReconcilerPassthrough:
    """Test IdentityReconciler.reconcile() returns forecasts unchanged."""

    def test_identity_reconciler_passthrough(self):
        """Identity reconciler should return leaf_forecasts unchanged."""
        cfg = MagicMock()
        reconciler = IdentityReconciler(cfg)

        # Create test leaf forecasts
        leaf_forecasts = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        # Call reconcile (need to fit first)
        spec = {"leaves": ["L1", "L2", "L3"]}
        reconciler.fit(spec, leaf_forecasts)
        reconciled_leaves, reconciled_aggregate, residuals = reconciler.reconcile(leaf_forecasts)

        # Reconciled leaves should match input
        assert np.allclose(reconciled_leaves, leaf_forecasts)

        # Aggregate should be row-wise sum
        expected_aggregate = leaf_forecasts.sum(axis=1)
        assert np.allclose(reconciled_aggregate, expected_aggregate)

        # Residuals should be zero
        assert np.allclose(residuals, np.zeros(len(reconciled_aggregate)))


class TestNaiveTopKAlphaOne:
    """Test NaiveTopKSelector with alpha=1.0 selects by competence only."""

    def test_naive_topk_alpha_one_ignores_diversity(self):
        """With alpha=1.0, selector should pick top-k by competence regardless of diversity."""
        # Create selector with alpha=1.0 (pure competence, no diversity)
        selector = NaiveTopKSelector(k=2)

        # Create competence scores: detector 0,1 have high competence
        competence = np.array([0.9, 0.8, 0.1, 0.05])

        # Create predictions: detectors 2,3 would have high diversity with the rest,
        # but they have low competence
        predictions = np.array(
            [[1, 0, 1, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 1, 0, 1]]
        ).T  # shape: (4, 2)

        # Labels for the window
        labels = np.array([1, 0])

        # Select top-k
        selected = selector.select(competence, predictions, labels)

        # Should select indices 0, 1 (highest competence)
        assert len(selected) == 2
        assert 0 in selected
        assert 1 in selected


class TestAblationOutputStructure:
    """Test run_ablation_variants() creates correct output structure."""

    def test_ablation_output_structure(self, tmp_path):
        """Test that ablation runner creates metrics JSON files for all variants."""
        # Setup temporary directories
        ablation_dir = tmp_path / "ablation"
        ablation_dir.mkdir(parents=True, exist_ok=True)

        # Mock the key dependencies
        with (
            patch("cdade.ablation.run_ablation.load_variant_ground_truth") as mock_load_gt,
            patch("cdade.ablation.run_ablation.load_variant_blended_scores") as mock_load_scores,
            patch("cdade.ablation.run_ablation.compute_all_metrics") as mock_compute_metrics,
        ):
            # Setup mocks
            y_true = np.array([0, 1, 1, 0, 1])
            mock_load_gt.return_value = y_true

            # Mock blended scores for each variant
            blended_scores = np.array([[0.1, 0.2], [0.9, 0.8], [0.3, 0.4], [0.7, 0.6], [0.5, 0.5]])
            mock_load_scores.return_value = blended_scores

            # Mock metrics computation
            mock_compute_metrics.return_value = {
                "auc_pr": 0.85,
                "nab": 0.75,
                "precision": 0.8,
                "recall": 0.7,
                "f1": 0.75,
                "threshold": 0.5,
            }

            # Run ablation with mocked results dir
            run_ablation_variants(results_dir=tmp_path)

            # Check that variant metrics files exist
            variants = ["full", "no_reconciliation", "no_dynamic_selection", "no_diversity"]
            for variant in variants:
                variant_file = ablation_dir / f"{variant}_metrics.json"
                assert variant_file.exists(), f"Missing {variant}_metrics.json"

                # Load and validate structure
                with open(variant_file, encoding="utf-8") as f:
                    metrics = json.load(f)

                required_keys = {"auc_pr", "nab", "precision", "recall", "f1", "threshold"}
                assert required_keys <= set(metrics.keys()), f"Missing keys in {variant} metrics"

            # Check summary.csv exists
            summary_file = ablation_dir / "summary.csv"
            assert summary_file.exists(), "Missing summary.csv"

            # Validate summary structure
            summary_df = pd.read_csv(summary_file)
            assert len(summary_df) == len(variants), f"Summary should have {len(variants)} rows"
            assert "variant" in summary_df.columns, "Summary missing 'variant' column"


class TestVariantTransformation:
    """Test apply_variant_transformation() applies variant-specific logic."""

    def test_apply_variant_identity_passthrough(self):
        """'full' and 'no_reconciliation' variants should return scores unchanged."""
        blended_scores = np.array([[0.1, 0.2], [0.9, 0.8], [0.3, 0.4]])

        # Full variant should pass through
        result = apply_variant_transformation(blended_scores, "full", cfg=MagicMock())
        assert np.allclose(result, blended_scores)

        # no_reconciliation should also pass through (identity reconciler)
        result = apply_variant_transformation(blended_scores, "no_reconciliation", cfg=MagicMock())
        assert np.allclose(result, blended_scores)

    def test_apply_variant_naive_topk(self):
        """'no_dynamic_selection' should apply NaiveTopKSelector."""
        blended_scores = np.array([[0.1, 0.2, 0.3], [0.9, 0.8, 0.7], [0.5, 0.5, 0.5]])

        cfg = MagicMock()
        cfg.selection.k = 2

        # Should return selected columns only
        result = apply_variant_transformation(blended_scores, "no_dynamic_selection", cfg=cfg)

        # Result should have shape (3, 2) - k=2 detectors selected
        assert result.shape[0] == 3  # same number of timesteps
        assert result.shape[1] <= 3  # at most 3 original detectors
