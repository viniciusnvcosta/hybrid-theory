"""Bottom-up hierarchical reconciliation.

Simple baseline: sum leaf-level forecasts, compare to aggregate.
"""

import numpy as np
import pandas as pd

from cdade.registry import register_reconciler


@register_reconciler("bottom_up")
class BottomUpReconciler:
    """Bottom-up hierarchical reconciliation (leaf → aggregate).

    Simple baseline: sum leaf-level forecasts, compare to aggregate.

    Args:
        cfg: Configuration object
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.S = None

    def fit(self, spec: dict, leaf_forecasts: pd.DataFrame) -> "BottomUpReconciler":
        """Build summing matrix from hierarchy spec.

        Args:
            spec: Hierarchy spec
            leaf_forecasts: Leaf-level forecasts (date × leaves)

        Returns:
            Self for chaining
        """
        self.spec = spec
        self.S = self.build_summing_matrix(spec)
        return self

    def build_summing_matrix(self, spec: dict) -> np.ndarray:
        """Build summing matrix (same as in summing_matrix.py)."""
        leaves = spec["leaves"]
        n_leaves = len(leaves)
        S = np.zeros((n_leaves, n_leaves), dtype=np.float64)
        S[0, :] = 1.0
        for i in range(n_leaves):
            S[i, i] = 1.0
        return S

    def reconcile(
        self,
        leaf_forecasts: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
        """Perform bottom-up reconciliation.

        Args:
            leaf_forecasts: Leaf-level forecasts (date index, leaf columns)

        Returns:
            Tuple of (reconciled_leaf_series, reconciled_aggregate_series, residuals)
        """
        # Leaf-level forecasts are already reconciled (identity)
        reconciled_leaves = leaf_forecasts.copy()

        # Aggregate = sum of leaves
        reconciled_aggregate = leaf_forecasts.sum(axis=1)

        # Residuals = actual - forecast
        residuals = reconciled_aggregate - leaf_forecasts.sum(axis=1)

        reconciled_leaf_series = reconciled_leaves
        reconciled_aggregate_series = reconciled_aggregate

        return reconciled_leaf_series, reconciled_aggregate_series, residuals
