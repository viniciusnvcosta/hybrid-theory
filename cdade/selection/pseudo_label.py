"""Pseudo-label generator for ensemble selection.

Generates pseudo-labels from detector scores via majority voting per window.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
from scipy.special import expit

logger = logging.getLogger(__name__)


def majority_vote_pseudo_labels(
    scores: np.ndarray,
    threshold: float = 0.5,
    method: Literal["soft", "hard"] = "soft",
) -> np.ndarray:
    """Generate pseudo-labels from detector scores via majority vote.

    For each window and series, uses the majority vote of detector scores
    to generate a pseudo-label for each detector.

    Args:
        scores: Anomaly scores from detectors.
            Shape: (n_windows, n_detectors, n_series)
        threshold: Score threshold for "anomalous" (default 0.5)
        method: "soft" (use sigmoid) or "hard" (use > threshold)

    Returns:
        pseudo_labels: Pseudo-labels per detector per window per series.
            Shape: (n_windows, n_detectors, n_series)

    Raises:
        ValueError: If scores shape is not (n_windows, n_detectors, n_series)
    """
    if scores.ndim != 3:
        raise ValueError(
            f"Expected scores shape (n_windows, n_detectors, n_series), got {scores.shape}"
        )

    n_windows, n_detectors, n_series = scores.shape
    pseudo_labels = np.zeros((n_windows, n_detectors, n_series), dtype=np.int8)

    for w in range(n_windows):
        for s in range(n_series):
            window_scores = scores[w, :, s]  # (n_detectors,)

            if method == "soft":
                # Soft voting: use sigmoid then majority
                sigmoid = expit(window_scores)
                pseudo_labels[w, :, s] = (sigmoid > threshold).astype(np.int8)
            else:
                # Hard voting: use threshold directly
                pseudo_labels[w, :, s] = (window_scores > threshold).astype(np.int8)

    logger.debug(
        f"Generated pseudo-labels: {pseudo_labels.shape} "
        f"(windows={n_windows}, detectors={n_detectors}, series={n_series})"
    )

    return pseudo_labels


def generate_windowed_labels(
    scores: np.ndarray,
    window_size: int = 12,
    stride: int = 1,
    threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate pseudo-labels for a sliding window over a single-detector score array.

    Each window label is 1 when the *mean* score in that window exceeds the threshold.

    Args:
        scores: Anomaly scores from a single detector.
            Shape: (n_timepoints, n_series)
        window_size: Size of sliding window (default 12)
        stride: Step between windows (default 1)
        threshold: Score threshold for pseudo-labels

    Returns:
        pseudo_labels: Pseudo-labels per window.
            Shape: (n_windows, n_series)
        windows: Start indices of each window.
            Shape: (n_windows,)
    """
    n_timepoints, n_series = scores.shape
    windows = np.arange(0, n_timepoints - window_size + 1, stride)
    n_windows = len(windows)

    pseudo_labels = np.zeros((n_windows, n_series), dtype=np.int8)

    for i, start in enumerate(windows):
        end = start + window_size
        avg_score = np.mean(scores[start:end, :], axis=0)
        pseudo_labels[i, :] = (avg_score > threshold).astype(np.int8)

    return pseudo_labels, windows
