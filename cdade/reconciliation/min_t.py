"""MinT (Minimum Trace) shrinkage reconciliation.

Solves: minimize trace(G'WG) subject to S'G = S (coherence).
"""

import numpy as np
import pandas as pd

from cdade.registry import register_reconciler


@register_reconciler("min_t")
class MinTReconciler:
    """MinT (Minimum Trace) shrinkage reconciliation.

    Solves: minimize trace(G'WG) subject to S'G = S (coherence).

    Args:
        cfg: Configuration object
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.S = None
        self.G = None

    def fit(self, spec: dict, leaf_forecasts: pd.DataFrame) -> "MinTReconciler":
        """Build summing matrix and solve MinT problem.

        Args:
            spec: Hierarchy spec
            leaf_forecasts: Leaf-level forecasts (date × leaves)

        Returns:
            Self for chaining
        """
        self.spec = spec
        self.S = self.build_summing_matrix(spec)
        self.solve_mint()
        return self

    def build_summing_matrix(self, spec: dict) -> np.ndarray:
        """Build summing matrix."""
        leaves = spec["leaves"]
        n_leaves = len(leaves)
        S = np.zeros((n_leaves, n_leaves), dtype=np.float64)
        S[0, :] = 1.0
        for i in range(n_leaves):
            S[i, i] = 1.0
        return S

    def solve_mint(self) -> np.ndarray:
        """Solve MinT problem using shrinkage covariance.

        G = (S'W⁻¹S)⁻¹S'W⁻¹
        """
        # Weight matrix W (diagonal precision matrix)
        W = np.eye(self.S.shape[1])

        # Compute (S'W⁻¹S)⁻¹S'W⁻¹
        S_T = self.S.T
        STWS = S_T @ (W @ self.S)
        STW_inv = np.linalg.inv(STWS)

        self.G = STW_inv @ S_T @ W
        self.STWS = STWS  # Store for testing

        # Verify coherence: S @ G @ S == S
        assert np.allclose(self.S @ self.G @ self.S, self.S, rtol=1e-6), "MinT coherence violation"

        return self.G

    def reconcile(self, leaf_forecasts: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
        """Perform MinT reconciliation.

        Args:
            leaf_forecasts: Leaf-level forecasts

        Returns:
            Tuple of (reconciled_leaf_series, reconciled_aggregate_series, residuals)
        """
        # Apply shrinkage G to leaf forecasts
        reconciled_leaves = leaf_forecasts.values @ self.G.T

        # Aggregate = sum of reconciled leaves
        reconciled_aggregate = reconciled_leaves.sum(axis=1)

        # Residuals
        residuals = reconciled_aggregate - reconciled_leaves.sum(axis=1)

        reconciled_leaf_series = pd.DataFrame(
            reconciled_leaves,
            index=leaf_forecasts.index,
        )
        reconciled_aggregate_series = pd.Series(
            reconciled_aggregate,
            name="aggregate",
        )

        return reconciled_leaf_series, reconciled_aggregate_series, residuals
