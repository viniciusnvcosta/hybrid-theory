"""Ensemble orchestrator module (L1→L2→L3).

Public API
----------
- CDADEOrchestrator: end-to-end pipeline runner
- log_experiment: MLflow logging of parameters, metrics, artifacts
"""

from cdade.ensemble.cdade import CDADEOrchestrator
from cdade.ensemble.logging import log_experiment

__all__ = ["CDADEOrchestrator", "log_experiment"]
