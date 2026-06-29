"""Base reconciler class for hierarchical reconciliation.

Provides abstract base class and utility functions.
"""

from abc import ABC, abstractmethod

import numpy as np


class BaseReconciler(ABC):
    """Abstract base class for hierarchical reconcilers.

    All reconcilers must implement:
    - build_summing_matrix(spec): Build S matrix from hierarchy spec
    - reconcile(leaf_forecasts): Apply reconciliation
    """

    @abstractmethod
    def build_summing_matrix(self, spec: dict) -> np.ndarray:
        """Build summing matrix from hierarchy spec.

        Args:
            spec: Hierarchy spec with 'leaves' key

        Returns:
            Summing matrix S
        """
        pass

    @abstractmethod
    def reconcile(
        self,
        leaf_forecasts: "np.ndarray",
    ) -> tuple["np.ndarray", "np.ndarray", "np.ndarray"]:
        """Perform reconciliation.

        Args:
            leaf_forecasts: Leaf-level forecasts

        Returns:
            Tuple of (reconciled_leaves, reconciled_aggregate, residuals)
        """
        pass
