# ABOUTME: Evaluation metrics for anomaly detection: AUC-PR, NAB, P/R/F1.
# ABOUTME: All functions take binary y_true and float scores as numpy arrays.

import numpy as np
from sklearn.metrics import average_precision_score, precision_score, recall_score


def compute_auc_pr(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Area under precision-recall curve via sklearn.metrics.average_precision_score.

    Args:
        y_true: Binary labels (0/1).
        scores: Anomaly scores (higher = more anomalous).

    Returns:
        AUC-PR value in [0, 1].
    """
    return float(average_precision_score(y_true, scores))


def compute_nab_score(y_true: np.ndarray, scores: np.ndarray, *, window: int = 4) -> float:
    """Simplified NAB streaming score.

    Credit: +1.0 if anomaly detected within [onset - window, onset + window].
    Penalty: -0.11 per false positive (NAB standard profile weights).
    Score normalised to [0, 1] by dividing by the perfect-detector score.
    Threshold: median of scores.

    Args:
        y_true: Binary labels.
        scores: Anomaly scores.
        window: Detection window half-width around anomaly onset.

    Returns:
        NAB score in [0, 1].
    """
    threshold = np.median(scores)
    predictions = (scores >= threshold).astype(int)

    # Find anomaly onsets (transitions from 0 to 1)
    anomaly_onsets = np.where(np.diff(np.concatenate(([0], y_true))) == 1)[0]

    score = 0.0
    fp_count = 0

    for idx, pred in enumerate(predictions):
        if pred == 1:  # Predicted anomaly
            is_tp = False
            # Check if within window of any anomaly onset
            for onset in anomaly_onsets:
                if abs(idx - onset) <= window:
                    is_tp = True
                    break
            if is_tp:
                score += 1.0
            else:
                fp_count += 1

    # Apply FP penalty
    penalty = 0.11 * fp_count
    score -= penalty

    # Normalize by perfect detector score (number of anomalies)
    perfect_score = float(len(anomaly_onsets))
    if perfect_score == 0:
        return 0.0

    normalized_score = score / perfect_score
    return float(np.clip(normalized_score, 0, 1))


def compute_pr_f1(
    y_true: np.ndarray, scores: np.ndarray, *, threshold: str = "median"
) -> dict[str, float]:
    """Precision, recall, F1 at the given threshold ('median' or float).

    Args:
        y_true: Binary labels.
        scores: Anomaly scores.
        threshold: Threshold value ('median' for median of scores) or float.

    Returns:
        Dict with keys 'precision', 'recall', 'f1'.
    """
    if threshold == "median":
        thresh_val = np.median(scores)
    else:
        thresh_val = float(threshold)

    predictions = (scores >= thresh_val).astype(int)

    # Handle edge case where all predictions are 0 or 1
    if np.sum(predictions) == 0:
        # No positive predictions
        precision = 0.0
        recall = 0.0
        f1 = 0.0
    else:
        precision = float(precision_score(y_true, predictions, zero_division=0))
        recall = float(recall_score(y_true, predictions, zero_division=0))
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def compute_all_metrics(
    y_true: np.ndarray, scores: np.ndarray, *, nab_window: int = 4
) -> dict[str, float]:
    """Compute and return {auc_pr, nab, precision, recall, f1, threshold}.

    Args:
        y_true: Binary labels.
        scores: Anomaly scores.
        nab_window: Detection window for NAB score.

    Returns:
        Dict with keys: auc_pr, nab, precision, recall, f1, threshold.
    """
    threshold = np.median(scores)
    auc_pr = compute_auc_pr(y_true, scores)
    nab = compute_nab_score(y_true, scores, window=nab_window)
    pr_f1_dict = compute_pr_f1(y_true, scores, threshold="median")

    return {
        "auc_pr": auc_pr,
        "nab": nab,
        "precision": pr_f1_dict["precision"],
        "recall": pr_f1_dict["recall"],
        "f1": pr_f1_dict["f1"],
        "threshold": float(threshold),
    }
