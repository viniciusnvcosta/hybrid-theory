"""Reconciliation module for hierarchical time series.

Provides bottom-up, MinT-shrink, EVT, and identity reconciliation methods.
"""

# Import identity reconciler to register it
from cdade.reconciliation.identity import IdentityReconciler
from cdade.registry import get_reconciler, register_reconciler

__all__ = [
    "register_reconciler",
    "get_reconciler",
    "IdentityReconciler",
]
