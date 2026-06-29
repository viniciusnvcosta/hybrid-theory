"""Reconciliation tests.

Tests for the hierarchical reconciliation module.
"""

import numpy as np
import pandas as pd
import pytest

from cdade.data.sivep import build_hierarchy_spec


@pytest.fixture
def sivep_hierarchy():
    """SIVEP hierarchy spec."""
    return build_hierarchy_spec()


@pytest.fixture
def sivep_forecasts():
    """Synthetic leaf forecasts for SIVEP."""
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=132, freq="MS")
    leaves = [
        "ARAGUAIA",
        "BAIXO AMAZONAS",
        "CARAJAS",
        "LAGO DE TUCURUI",
        "MARAJO I",
        "MARAJO II",
        "METROPOLITANA I",
        "METROPOLITANA II",
        "METROPOLITANA III",
        "RIO CAETES",
        "TAPAJOS",
        "TOCANTINS",
        "XINGU",
    ]
    return pd.DataFrame(
        np.random.randint(0, 50, (132, 13)),
        index=dates,
        columns=leaves,
    )


def test_bottom_up_coherence(sivep_hierarchy, sivep_forecasts):
    """Test that bottom-up reconciliation maintains coherence."""
    from cdade.reconciliation import bottom_up

    reconciler = bottom_up.BottomUpReconciler(None)
    reconciler.fit(sivep_hierarchy, sivep_forecasts)

    reconciled_leaves, reconciled_aggregate, residuals = reconciler.reconcile(sivep_forecasts)

    # Check shape
    assert reconciled_leaves.shape == sivep_forecasts.shape

    # Check coherence: leaves sum to aggregate (within rounding)
    leaf_sum = reconciled_leaves.sum(axis=1)
    aggregate = reconciled_aggregate

    # Tolerance for integer rounding
    assert np.allclose(leaf_sum, aggregate, atol=2)

    # Check no NaN
    assert not reconciled_leaves.isna().any().any()
    assert not pd.isna(aggregate).any()


def test_mint_unbiasedness(sivep_hierarchy, sivep_forecasts):
    """Test that MinT satisfies S @ G @ S == S."""
    from cdade.reconciliation import min_t

    reconciler = min_t.MinTReconciler(None)
    reconciler.fit(sivep_hierarchy, sivep_forecasts)

    # Build S matrix
    S = reconciler.build_summing_matrix(sivep_hierarchy)

    # Check coherence property
    assert np.allclose(
        S @ reconciler.G @ S,
        S,
        rtol=1e-6,
    )


def test_mint_invertible(sivep_hierarchy):
    """Test that MinT solution is unique (invertible)."""
    from cdade.reconciliation import min_t

    reconciler = min_t.MinTReconciler(None)
    reconciler.fit(sivep_hierarchy, sivep_forecasts)

    # Check that STWS is invertible (full rank)
    assert np.linalg.matrix_rank(reconciler.STWS) == reconciler.STWS.shape[0]
