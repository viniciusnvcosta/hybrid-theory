"""Baseline methods for CDADE comparison.

Provides implementations of comparison baselines for anomaly detection:
- B1: Farrington/Noufaily (epidemiological anomaly detection)
- B2: Best single detector
- B3: Full ensemble average
- B4: Static Top-K (Eze et al.)
- B5: Reconciliation + EVT (Kandanaarachchi)

Author: CDADE project
"""

from .farrington import FarringtonDetector, FarringtonConfig, register_baseline_detector

__all__ = [
    "FarringtonDetector",
    "FarringtonConfig",
    "register_baseline_detector",
]