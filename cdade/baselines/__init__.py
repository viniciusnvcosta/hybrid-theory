"""Baseline methods for CDADE comparison.

Provides implementations of comparison baselines for anomaly detection:
- B1: Farrington/Noufaily (epidemiological anomaly detection)
- B2: Best single detector
- B3: Full ensemble average (AOM)
- B4: Static Top-K greedy set-cover (Eze et al.)
- B5: Reconciliation + EVT (Kandanaarachchi)

Author: CDADE project
"""

from .ensemble_average import EnsembleAverageConfig, EnsembleAverageDetector
from .farrington import FarringtonConfig, FarringtonDetector, register_baseline_detector
from .reconciliation_evt import ReconciliationEVTConfig, ReconciliationEVTDetector
from .static_topk import StaticTopKConfig, StaticTopKDetector

__all__ = [
    "FarringtonDetector",
    "FarringtonConfig",
    "register_baseline_detector",
    "EnsembleAverageDetector",
    "EnsembleAverageConfig",
    "StaticTopKDetector",
    "StaticTopKConfig",
    "ReconciliationEVTDetector",
    "ReconciliationEVTConfig",
]
