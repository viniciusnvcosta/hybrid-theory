"""Reconciliation module for hierarchical time series.

Provides bottom-up, MinT-shrink, EVT, and identity reconciliation methods.
"""

# Import all reconcilers to register them
from cdade.reconciliation.bottom_up import BottomUpReconciler
from cdade.reconciliation.evt import EVTReconciler
from cdade.reconciliation.identity import IdentityReconciler
from cdade.reconciliation.min_t import MinTReconciler
from cdade.registry import get_reconciler, register_reconciler

__all__ = [
    "register_reconciler",
    "get_reconciler",
    "BottomUpReconciler",
    "EVTReconciler",
    "IdentityReconciler",
    "MinTReconciler",
]
