"""Static Top-K greedy set-cover baseline (B4) for CDADE comparison.

Selects a fixed subset of k detectors on the validation set using a greedy
set-cover strategy that rewards both individual competence and pairwise
diversity.  The selected subset is then frozen for test-time inference,
mirroring the incumbent method described in:

Reference:
    Eze, J. et al. (2023). Anomaly Detection in Hierarchical Time Series.

Author: CDADE project
"""

import logging
from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import f1_score

from cdade.baselines.single_best import create_detector
from cdade.registry import get_detector, list_detectors, register_baseline_detector

logger = logging.getLogger(__name__)


@dataclass
class StaticTopKConfig:
    """Configuration for the static top-k greedy selector.

    Attributes:
        k: Number of detectors to select. When 0, uses sqrt(n_detectors).
        alpha: Weight balancing competence (1.0) vs diversity (0.0).
        detector_names: Detector pool. Uses all registered when empty.
        normalize: Whether to normalise scores before aggregation.
    """

    k: int = 0
    alpha: float = 0.5
    detector_names: list[str] = field(default_factory=list)
    normalize: bool = True


def _normalize_scores(scores: np.ndarray) -> np.ndarray:
    lo, hi = scores.min(), scores.max()
    if hi == lo:
        return np.zeros_like(scores)
    return (scores - lo) / (hi - lo)


def _binary_predictions(scores: np.ndarray) -> np.ndarray:
    """Threshold at score median to obtain binary labels."""
    return (scores >= np.median(scores)).astype(int)


def _f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, zero_division=0))


def greedy_set_cover_selection(
    val_scores: np.ndarray,
    val_labels: np.ndarray,
    k: int,
    alpha: float,
) -> list[int]:
    """Greedy set-cover: iteratively add the detector that maximises
    α·ΔCompetence + (1-α)·ΔDiversity over the already-selected set.

    Args:
        val_scores: Detector scores on validation set
            (shape: [n_detectors, n_val]).
        val_labels: Ground-truth binary labels (shape: [n_val]).
        k: Number of detectors to select.
        alpha: Competence weight ∈ [0, 1].

    Returns:
        Indices of selected detectors (length k).
    """
    n_det = val_scores.shape[0]
    k = min(k, n_det)

    # Pre-compute per-detector binary predictions and F1 on validation set
    preds = np.stack([_binary_predictions(val_scores[i]) for i in range(n_det)], axis=0)
    competences = np.array([_f1(val_labels, preds[i]) for i in range(n_det)])

    selected: list[int] = []
    remaining = list(range(n_det))

    # Diversity: mean pairwise agreement (lower = more diverse)
    def _pairwise_diversity(idx: int, current: list[int]) -> float:
        if not current:
            return 1.0  # first selection: maximum diversity bonus
        agreements = [np.mean(preds[idx] == preds[j]) for j in current]
        return 1.0 - float(np.mean(agreements))

    for _ in range(k):
        if not remaining:
            break
        best_idx = -1
        best_score = -np.inf
        for idx in remaining:
            div = _pairwise_diversity(idx, selected)
            combined = alpha * competences[idx] + (1.0 - alpha) * div
            if combined > best_score:
                best_score = combined
                best_idx = idx
        selected.append(best_idx)
        remaining.remove(best_idx)

    return selected


class StaticTopKDetector:
    """Static Top-K greedy set-cover detector (B4).

    Selects a fixed detector subset on the validation set and averages their
    scores at test time.  No re-selection occurs after validation.

    Attributes:
        config: Detector configuration.
        selected_indices: Indices into the detector pool chosen at fit time.
        fitted_detectors: All fitted detectors (pool); selection applies at
            score time.
    """

    def __init__(self, config: StaticTopKConfig | None = None) -> None:
        """Initialise static top-k detector.

        Args:
            config: Configuration object; uses defaults when None.
        """
        self.config = config or StaticTopKConfig()
        self.selected_indices: list[int] = []
        self.fitted_detectors: list[tuple[str, object]] = []
        self._pool_names: list[str] = []

    def _resolve_names(self) -> list[str]:
        return self.config.detector_names if self.config.detector_names else list_detectors()

    def fit(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> "StaticTopKDetector":
        """Fit the detector pool and select the top-k subset.

        Args:
            X_train: Training data (shape: [n_train, n_features]).
            X_val: Validation data (shape: [n_val, n_features]).
            y_val: Validation binary labels (shape: [n_val]).

        Returns:
            Self for chaining.
        """
        self._pool_names = self._resolve_names()
        self.fitted_detectors = []

        for name in self._pool_names:
            try:
                cls = get_detector(name)
                detector = create_detector(cls)
                detector.fit(X_train)
                self.fitted_detectors.append((name, detector))
                logger.debug("Fitted detector '%s'", name)
            except Exception as exc:
                logger.warning("Skipping detector '%s': %s", name, exc)

        if not self.fitted_detectors:
            raise RuntimeError("No detectors could be fitted.")

        # Collect validation scores
        val_score_list: list[np.ndarray] = []
        valid_names: list[str] = []
        for name, det in self.fitted_detectors:
            try:
                s = det.score(X_val)
                if self.config.normalize:
                    s = _normalize_scores(s)
                val_score_list.append(s)
                valid_names.append(name)
            except Exception as exc:
                logger.warning("Detector '%s' failed at val score: %s", name, exc)

        if not val_score_list:
            raise RuntimeError("All detectors failed on validation set.")

        val_scores = np.stack(val_score_list, axis=0)  # [n_detectors, n_val]

        # Determine k
        k = self.config.k if self.config.k > 0 else max(1, int(np.sqrt(len(val_score_list))))

        self.selected_indices = greedy_set_cover_selection(val_scores, y_val, k, self.config.alpha)

        selected_names = [valid_names[i] for i in self.selected_indices]
        logger.info(
            "StaticTopK selected %d / %d detectors: %s",
            len(self.selected_indices),
            len(self.fitted_detectors),
            selected_names,
        )
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Compute averaged scores of the selected detector subset.

        Args:
            X: Data to score (shape: [n_samples, n_features]).

        Returns:
            Averaged anomaly scores (shape: [n_samples]).
        """
        if not self.selected_indices:
            raise RuntimeError("Detector must be fitted before scoring.")

        selected_scores: list[np.ndarray] = []
        detectors = [
            (name, det)
            for i, (name, det) in enumerate(self.fitted_detectors)
            if i in self.selected_indices
        ]

        for name, det in detectors:
            try:
                s = det.score(X)
                if self.config.normalize:
                    s = _normalize_scores(s)
                selected_scores.append(s)
            except Exception as exc:
                logger.warning("Detector '%s' failed at test score: %s", name, exc)

        if not selected_scores:
            raise RuntimeError("All selected detectors failed during scoring.")

        return np.mean(np.stack(selected_scores, axis=0), axis=0)

    def predict(self, X: np.ndarray, threshold: float | None = None) -> np.ndarray:
        """Binary anomaly predictions.

        Args:
            X: Data (shape: [n_samples, n_features]).
            threshold: Score threshold. Defaults to score median.

        Returns:
            Binary predictions (shape: [n_samples]).
        """
        scores = self.score(X)
        thresh = threshold if threshold is not None else float(np.median(scores))
        return (scores >= thresh).astype(int)


@register_baseline_detector("static_topk")
class RegisteredStaticTopKDetector(StaticTopKDetector):
    """Static top-k greedy detector registered for baseline comparison."""

    pass
