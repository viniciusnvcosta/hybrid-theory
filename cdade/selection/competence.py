"""Competence estimator for ensemble selection (META-DES).

Implements local accuracy estimation following Cruz et al. (2015).
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def meta_des_competence(
    pseudo_labels: np.ndarray,
    true_labels: np.ndarray,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Compute competence scores for each detector using META-DES.

    Implements local accuracy estimator:
    C_i(w) = (1/(2m)) * (precision_i(w) + recall_i(w))

    Args:
        pseudo_labels: Pseudo-labels from detector scores.
            Shape: (n_windows, n_detectors, n_series)
        true_labels: Ground-truth labels.
            Shape: (n_windows, n_series) or (n_windows, n_detectors, n_series)
        weights: Optional per-window weights.
            Shape: (n_windows,) or (n_windows, n_detectors, n_series)

    Returns:
        competence: Competence scores per detector per window per series.
            Shape: (n_windows, n_detectors, n_series)

    Raises:
        ValueError: If shapes are incompatible
    """
    # Ensure true_labels has detector dimension
    if true_labels.ndim == 2:
        # Broadcast single label per timepoint to detector dimension
        true_labels = np.tile(true_labels[:, np.newaxis, :], (1, pseudo_labels.shape[1], 1))

    if pseudo_labels.shape != true_labels.shape:
        raise ValueError(
            f"pseudolabels shape {pseudo_labels.shape} "
            f"must match truelabels shape {true_labels.shape}"
        )

    n_windows, n_detectors, n_series = pseudo_labels.shape

    competence = np.zeros((n_windows, n_detectors, n_series))

    for w in range(n_windows):
        for d in range(n_detectors):
            for s in range(n_series):
                # Binary labels
                pred = pseudo_labels[w, d, s]  # 0 or 1
                true = true_labels[w, d, s]  # 0 or 1

                if np.sum(true) == 0:  # Avoid division by zero
                    competence[w, d, s] = 1.0
                    continue

                # Precision = TP / (TP + FP)
                tp = np.sum((pred == 1) & (true == 1))
                fp = np.sum((pred == 1) & (true == 0))

                # Recall = TP / (TP + FN)
                tp = np.sum((pred == 1) & (true == 1))
                fn = np.sum((pred == 0) & (true == 1))

                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

                competence[w, d, s] = 0.5 * (precision + recall)

    logger.debug(
        f"Computed competence: {competence.shape} "
        f"(windows={n_windows}, detectors={n_detectors}, series={n_series})"
    )

    return competence


def windowed_competence(
    scores: np.ndarray,
    true_labels: np.ndarray,
    window_size: int = 12,
    stride: int = 1,
    threshold: float = 0.5,
) -> np.ndarray:
    """Compute competence scores over sliding windows.

    Args:
        scores: Anomaly scores from single detector.
            Shape: (n_timepoints, n_series)
        true_labels: Ground-truth labels.
            Shape: (n_timepoints, n_series)
        window_size: Size of sliding window
        stride: Step between windows
        threshold: Score threshold for pseudo-labels

    Returns:
        competence: Competence per window per series.
            Shape: (n_windows, n_series)
    """
    n_timepoints, n_series = scores.shape
    n_windows = (n_timepoints - window_size) // stride + 1

    # Generate pseudo-labels
    pseudo_labels = np.zeros((n_windows, n_series), dtype=np.int8)
    windows = np.arange(0, n_timepoints - window_size + 1, stride)

    for i, start in enumerate(windows):
        end = start + window_size
        window_scores = scores[start:end, :]

        # Use mean score as pseudo-label
        window_scores = np.mean(window_scores, axis=0)
        pseudo_labels[i, :] = (window_scores > threshold).astype(np.int8)

    # Compute competence
    competence = meta_des_competence(
        pseudo_labels[:, np.newaxis, :],
        true_labels[windows, :],
    )

    return competence[:, 0, :]  # Remove detector dimension
