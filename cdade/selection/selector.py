"""Subset selector for dynamic ensemble selection (L3).

Implements K*(w) = argmax_{K} [ alpha * C_K(w) + (1-alpha) * D_K(w) ]

following the META-DES framework (Cruz et al., 2015) extended with
Q-statistic diversity weighting.
"""

from __future__ import annotations

import logging
from itertools import combinations

import numpy as np

from cdade.registry import register_selector
from cdade.selection.diversity import ensemble_q_diversity

logger = logging.getLogger(__name__)


@register_selector("meta_des")
class MetaDESSelector:
    """Dynamic ensemble selector using META-DES competence + Q-diversity.

    At each window w:
      1. Rank detectors by competence C_i(w) ∈ [0,1].
      2. For candidate subsets K of size k, compute the blended score:
             score(K) = alpha * mean_C(K) + (1-alpha) * D_Q(K)
      3. Return K*(w) = argmax score(K).

    For pools where exhaustive search over all C(n,k) subsets is too costly,
    a greedy top-k-by-competence seed is used and diversity is only evaluated
    for small perturbations (swap-one).

    Args:
        k: Number of detectors to select per window.
        alpha: Trade-off weight ∈ [0,1]. alpha=1 → pure competence,
               alpha=0 → pure diversity.
        exhaustive_limit: Max pool size for exhaustive C(n,k) search.
               Above this, uses greedy + swap-one.
    """

    def __init__(self, k: int = 5, alpha: float = 0.5, exhaustive_limit: int = 15) -> None:
        self.k = k
        self.alpha = alpha
        self.exhaustive_limit = exhaustive_limit

    def select(
        self,
        competence: np.ndarray,
        predictions: np.ndarray,
        labels: np.ndarray,
    ) -> np.ndarray:
        """Select the best detector subset for one window.

        Args:
            competence: Competence score per detector.
                Shape: (n_detectors,)
            predictions: Binary predictions per detector over the window.
                Shape: (n_detectors, n_timepoints)
            labels: Ground-truth labels over the window.
                Shape: (n_timepoints,)

        Returns:
            selected: Indices of the selected k detectors.
                Shape: (k,)
        """
        n_detectors = len(competence)
        k = min(self.k, n_detectors)

        if n_detectors == 1:
            return np.array([0])

        if n_detectors <= self.exhaustive_limit:
            return self._exhaustive_select(competence, predictions, labels, k)
        return self._greedy_select(competence, predictions, labels, k)

    def _score_subset(
        self,
        indices: list[int],
        competence: np.ndarray,
        predictions: np.ndarray,
        labels: np.ndarray,
    ) -> float:
        """Compute blended score for a subset of detector indices."""
        mean_c = float(competence[indices].mean())
        if len(indices) < 2:
            diversity = 0.0
        else:
            diversity = ensemble_q_diversity(predictions[indices], labels)
        return self.alpha * mean_c + (1.0 - self.alpha) * diversity

    def _exhaustive_select(
        self,
        competence: np.ndarray,
        predictions: np.ndarray,
        labels: np.ndarray,
        k: int,
    ) -> np.ndarray:
        """Exhaustive search over all C(n, k) subsets."""
        best_score = -np.inf
        best_subset: list[int] = list(range(k))

        for subset in combinations(range(len(competence)), k):
            score = self._score_subset(list(subset), competence, predictions, labels)
            if score > best_score:
                best_score = score
                best_subset = list(subset)

        logger.debug(
            f"Exhaustive selection: k={k}, best_score={best_score:.4f}, subset={best_subset}"
        )
        return np.array(best_subset)

    def _greedy_select(
        self,
        competence: np.ndarray,
        predictions: np.ndarray,
        labels: np.ndarray,
        k: int,
    ) -> np.ndarray:
        """Greedy seed (top-k by competence) + swap-one improvement."""
        # Seed: top-k detectors by competence
        seed = list(np.argsort(competence)[::-1][:k])
        best_score = self._score_subset(seed, competence, predictions, labels)

        improved = True
        while improved:
            improved = False
            excluded = [i for i in range(len(competence)) if i not in seed]
            for out_idx, in_idx in ((o, i) for o in seed for i in excluded):
                candidate = [i if i != out_idx else in_idx for i in seed]
                score = self._score_subset(candidate, competence, predictions, labels)
                if score > best_score + 1e-9:
                    best_score = score
                    seed = candidate
                    improved = True
                    break  # restart scan after any improvement

        logger.debug(f"Greedy+swap selection: k={k}, best_score={best_score:.4f}, subset={seed}")
        return np.array(sorted(seed))


@register_selector("naive_topk")
class NaiveTopKSelector:
    """Baseline selector: always pick the k detectors with highest competence.

    Ignores diversity — equivalent to MetaDESSelector with alpha=1.

    Args:
        k: Number of detectors to select.
    """

    def __init__(self, k: int = 5) -> None:
        self.k = k

    def select(
        self,
        competence: np.ndarray,
        predictions: np.ndarray,
        labels: np.ndarray,
    ) -> np.ndarray:
        """Select top-k detectors by competence.

        Args:
            competence: Competence score per detector. Shape: (n_detectors,)
            predictions: Not used, kept for interface compatibility.
            labels: Not used, kept for interface compatibility.

        Returns:
            selected: Indices of top-k detectors. Shape: (k,)
        """
        k = min(self.k, len(competence))
        return np.argsort(competence)[::-1][:k]
