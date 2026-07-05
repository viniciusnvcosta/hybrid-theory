# ABOUTME: Identity reconciler for ablation studies.
# ABOUTME: Passes input forecasts through unchanged (no-op for testing).

import numpy as np

from cdade.registry import register_reconciler


@register_reconciler("identity")
class IdentityReconciler:
    """Identity reconciler for ablation studies.

    Passes forecasts through unchanged (no-op). Used to test L2 contribution in isolation.

    Args:
        cfg: Configuration object
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.S = None

    def fit(self, spec: dict, leaf_forecasts) -> "IdentityReconciler":
        """Build summing matrix (identity reconciler doesn't use it).

        Args:
            spec: Hierarchy spec
            leaf_forecasts: Leaf-level forecasts

        Returns:
            Self for chaining
        """
        self.spec = spec
        return self

    def build_summing_matrix(self, spec: dict) -> np.ndarray:
        """Build summing matrix (identity reconciler doesn't use it).

        Args:
            spec: Hierarchy spec

        Returns:
            Identity matrix
        """
        leaves = spec["leaves"]
        n_leaves = len(leaves)
        return np.eye(n_leaves, dtype=np.float64)

    def reconcile(
        self,
        leaf_forecasts,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Perform identity reconciliation.

        Args:
            leaf_forecasts: Leaf-level forecasts

        Returns:
            Tuple of (reconciled_leaves, reconciled_aggregate, residuals)
        """
        # Identity: return forecasts unchanged
        reconciled_leaves = leaf_forecasts

        # Aggregate = sum of leaves (same as input)
        reconciled_aggregate = leaf_forecasts.sum(axis=1)

        # Residuals = actual - forecast = 0 for identity
        residuals = np.zeros(len(reconciled_aggregate), dtype=np.float64)

        return reconciled_leaves, reconciled_aggregate, residuals
