"""Diversity measures for ensemble selection.

Implements the Q-statistic pairwise diversity following Kuncheva & Whitaker (2003).
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def q_statistic_pair(
    y1: np.ndarray,
    y2: np.ndarray,
    labels: np.ndarray,
) -> float:
    """Compute Q-statistic for a pair of binary classifiers.

    Q = (N11*N00 - N01*N10) / (N11*N00 + N01*N10)

    where Nab = #{samples where classifier 1 is correct=a, classifier 2 is correct=b}.

    Q ∈ [-1, 1]. Positive values indicate agreement (correlated errors),
    negative values indicate diversity (complementary errors).

    Args:
        y1: Binary predictions from classifier 1. Shape: (n_samples,)
        y2: Binary predictions from classifier 2. Shape: (n_samples,)
        labels: Ground-truth binary labels. Shape: (n_samples,)

    Returns:
        Q-statistic value in [-1, 1].
    """
    correct1 = (y1 == labels).astype(int)
    correct2 = (y2 == labels).astype(int)

    n11 = np.sum((correct1 == 1) & (correct2 == 1))
    n10 = np.sum((correct1 == 1) & (correct2 == 0))
    n01 = np.sum((correct1 == 0) & (correct2 == 1))
    n00 = np.sum((correct1 == 0) & (correct2 == 0))

    denom = n11 * n00 + n01 * n10
    if denom == 0:
        return 0.0

    return float((n11 * n00 - n01 * n10) / denom)


def ensemble_q_diversity(
    predictions: np.ndarray,
    labels: np.ndarray,
) -> float:
    """Compute mean pairwise Q-statistic diversity for an ensemble.

    D_Q = 1 - (2 / (L*(L-1))) * sum_{i<j} Q(i, j)

    Low Q (near -1) = diverse; high Q (near 1) = redundant.
    The returned value D_Q ∈ [0, 1] is 1 - normalised_Q so that
    higher means more diverse (matches the α-blend convention).

    Args:
        predictions: Binary predictions from all detectors.
            Shape: (n_detectors, n_samples)
        labels: Ground-truth binary labels. Shape: (n_samples,)

    Returns:
        Diversity score D_Q ∈ [0, 1], higher = more diverse.

    Raises:
        ValueError: If fewer than 2 detectors are provided.
    """
    n_detectors = predictions.shape[0]
    if n_detectors < 2:
        raise ValueError(f"Need at least 2 detectors for Q-statistic, got {n_detectors}")

    q_sum = 0.0
    n_pairs = 0
    for i in range(n_detectors):
        for j in range(i + 1, n_detectors):
            q_sum += q_statistic_pair(predictions[i], predictions[j], labels)
            n_pairs += 1

    mean_q = q_sum / n_pairs  # in [-1, 1]
    # Convert: high Q (correlated) → low diversity; low/neg Q → high diversity
    diversity = 0.5 * (1.0 - mean_q)  # maps [-1,1] → [1,0] then rescale to [0,1]
    return float(np.clip(diversity, 0.0, 1.0))


def windowed_diversity(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5,
) -> float:
    """Compute Q-diversity over a single window of detector scores.

    Args:
        scores: Anomaly scores per detector.
            Shape: (n_detectors, n_timepoints)
        labels: Ground-truth labels.
            Shape: (n_timepoints,)
        threshold: Score → binary threshold.

    Returns:
        Diversity score D_Q ∈ [0, 1].
    """
    predictions = (scores > threshold).astype(int)
    return ensemble_q_diversity(predictions, labels.astype(int))
