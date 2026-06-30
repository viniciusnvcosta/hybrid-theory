"""Drift detectors for competence reset in dynamic ensemble selection.

Wraps `river` ADWIN and Page-Hinkley detectors. When drift is detected,
the caller should reset detector competence scores for the affected window.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
from river.drift import ADWIN, PageHinkley

logger = logging.getLogger(__name__)


class DriftDetector:
    """Sliding-window drift detector backed by ADWIN or Page-Hinkley.

    Feed one value per window via `update(value)`. The detector signals
    drift by setting `self.drift_detected = True` after that call.

    Args:
        method: "adwin" or "page_hinkley".
        delta: ADWIN confidence parameter (default 0.002).
        min_instances: ADWIN minimum window instances (default 30).
        threshold: Page-Hinkley detection threshold (default 50.0).
        alpha: Page-Hinkley forgetting factor (default 1 - 0.0001).
    """

    def __init__(
        self,
        method: Literal["adwin", "page_hinkley"] = "adwin",
        *,
        delta: float = 0.002,
        min_instances: int = 30,
        threshold: float = 50.0,
        alpha: float = 1 - 0.0001,
    ) -> None:
        self.method = method
        self._drift_count = 0

        if method == "adwin":
            self._detector = ADWIN(delta=delta)
        elif method == "page_hinkley":
            self._detector = PageHinkley(
                min_instances=min_instances, threshold=threshold, alpha=alpha
            )
        else:
            raise ValueError(f"Unknown drift method '{method}'. Use 'adwin' or 'page_hinkley'.")

        self.drift_detected: bool = False

    def update(self, value: float) -> bool:
        """Feed one observation and check for drift.

        Args:
            value: Scalar value (e.g., mean competence, error rate).

        Returns:
            True if drift was detected at this step.
        """
        self._detector.update(value)
        self.drift_detected = self._detector.drift_detected
        if self.drift_detected:
            self._drift_count += 1
            logger.info(f"Drift detected by {self.method} (total={self._drift_count})")
        return self.drift_detected

    def reset(self) -> None:
        """Manually reset the detector state."""
        if self.method == "adwin":
            self._detector = ADWIN(delta=self._detector.delta)
        else:
            self._detector = PageHinkley(
                min_instances=self._detector.min_instances,
                threshold=self._detector.threshold,
                alpha=self._detector.alpha,
            )
        self.drift_detected = False

    @property
    def n_detections(self) -> int:
        """Total number of drift events detected."""
        return self._drift_count


def scan_for_drift(
    signal: np.ndarray,
    method: Literal["adwin", "page_hinkley"] = "adwin",
    **kwargs: object,
) -> tuple[np.ndarray, int]:
    """Scan a 1-D signal and return per-step drift flags.

    Useful for offline analysis of competence trajectories.

    Args:
        signal: 1-D array of values to scan. Shape: (n_steps,)
        method: Drift method ("adwin" or "page_hinkley").
        **kwargs: Forwarded to DriftDetector constructor.

    Returns:
        flags: Boolean array, True where drift was detected. Shape: (n_steps,)
        n_detections: Total number of drift events.
    """
    detector = DriftDetector(method=method, **kwargs)
    flags = np.zeros(len(signal), dtype=bool)
    for t, val in enumerate(signal):
        flags[t] = detector.update(float(val))
    return flags, detector.n_detections
