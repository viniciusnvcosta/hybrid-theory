"""Detector module for unsupervised anomaly detection.

Provides PyOD wrappers and scratch implementations for the base detector pool.
"""

from cdade.registry import register_detector

__all__ = [
    "register_detector",
    "get_detector",
]
