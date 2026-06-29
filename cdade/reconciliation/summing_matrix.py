"""Summing matrix builder for hierarchical reconciliation.

Provides utility function to build S matrix from hierarchy specification.
"""

import numpy as np


def build_summing_matrix(spec: dict) -> np.ndarray:
    """Build summing matrix S from hierarchy spec.

    Args:
        spec: Hierarchy spec from data loader:
            {
                "state": "PA",
                "leaves": ["HR1", "HR2", ..., "HR13"]
            }

    Returns:
        Summing matrix S of shape (n_leaves + 1, n_leaves):
        - Row 0: all ones (aggregate = sum of leaves)
        - Rows 1..n: identity (leaves are atomic)
    """
    leaves = spec["leaves"]
    n_leaves = len(leaves)

    S = np.zeros((n_leaves + 1, n_leaves), dtype=np.float64)

    # Row 0: aggregate = sum of leaves
    S[0, :] = 1.0

    # Rows 1..n: identity mapping
    for i in range(n_leaves):
        S[i, i] = 1.0

    return S
