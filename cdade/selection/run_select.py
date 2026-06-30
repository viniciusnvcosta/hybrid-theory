"""CLI entry point for the selection pipeline stage (DVC: select).

Reads reconciled anomaly scores from results/reconciliation/,
runs per-window META-DES selection with optional drift detection,
and writes the active-subset indices + blended scores to results/selection/.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


@hydra.main(config_path="../../configs", config_name="config", version_base="1.2")
def main(cfg: DictConfig) -> None:
    """Run dynamic ensemble selection over reconciled detector scores."""
    from cdade.registry import get_selector
    from cdade.selection import (
        DriftDetector,
        majority_vote_pseudo_labels,
        meta_des_competence,
    )

    recon_dir = Path("results/reconciliation")
    out_dir = Path("results/selection")
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- load reconciled scores ------------------------------------------
    scores_path = recon_dir / "leaf_forecasts_reconciled.csv"
    if not scores_path.exists():
        raise FileNotFoundError(f"Reconciled scores not found: {scores_path}")

    scores_df = pd.read_csv(scores_path, index_col=0)
    # Shape expected after reconciliation: (n_timepoints, n_series)
    scores = scores_df.values.astype(np.float64)
    n_timepoints, n_series = scores.shape
    logger.info(f"Loaded scores: {scores.shape}")

    # --- sliding-window setup -------------------------------------------
    window_size: int = cfg.selection.window
    stride: int = getattr(cfg.selection, "stride", 1)
    alpha: float = cfg.selection.alpha
    k: int = getattr(cfg.selection, "k", 5)
    threshold: float = getattr(cfg.selection, "threshold", 0.5)

    windows = np.arange(0, n_timepoints - window_size + 1, stride)
    n_windows = len(windows)
    logger.info(f"Windows: {n_windows} (size={window_size}, stride={stride})")

    # For this stage scores is (n_timepoints, n_series); treat each series as
    # one "detector" for the selection step.  The full multi-detector path
    # is wired in the ensemble orchestrator (Stage 5).
    n_detectors = n_series
    windowed_scores = np.stack(
        [scores[s : s + window_size, :].T for s in windows],
        axis=0,
    )  # (n_windows, n_detectors, window_size)

    # --- pseudo-labels --------------------------------------------------
    # Reshape to (n_windows, n_detectors, 1) so the helper gets a 3-D array
    score_summary = windowed_scores.mean(axis=2)[:, :, np.newaxis]
    pseudo_labels = majority_vote_pseudo_labels(
        score_summary, threshold=threshold
    )  # (n_windows, n_detectors, 1)
    pseudo_labels_flat = pseudo_labels[:, :, 0]  # (n_windows, n_detectors)

    # --- competence -------------------------------------------------------
    # Without true labels we fall back to self-agreement competence:
    # each detector is compared to the majority vote of the full pool.
    pool_vote = (pseudo_labels_flat.mean(axis=1, keepdims=True) > 0.5).astype(
        np.int8
    )  # (n_windows, 1)
    pool_vote_broadcast = np.tile(pool_vote, (1, n_detectors))  # (n_windows, n_detectors)

    competence = meta_des_competence(
        pseudo_labels_flat[:, :, np.newaxis].astype(np.int8),
        pool_vote_broadcast[:, :, np.newaxis].astype(np.int8),
    )[:, :, 0]  # (n_windows, n_detectors)

    # --- drift detection -----------------------------------------------
    drift_method: str = getattr(cfg.selection, "drift_method", "adwin")
    drift_detector = DriftDetector(method=drift_method)
    drift_flags = np.zeros(n_windows, dtype=bool)
    for w in range(n_windows):
        mean_comp = float(competence[w].mean())
        drift_flags[w] = drift_detector.update(mean_comp)
        if drift_flags[w]:
            # Reset competence for this window to uniform
            competence[w] = 0.5

    logger.info(f"Drift events detected: {drift_detector.n_detections}")

    # --- selector -------------------------------------------------------
    selector_name: str = cfg.selection.name
    selector = get_selector(selector_name)(k=k, alpha=alpha)

    selected_indices = np.zeros((n_windows, k), dtype=int)
    for w in range(n_windows):
        # predictions: (n_detectors, window_size)
        preds = (windowed_scores[w] > threshold).astype(int)
        # majority label per timepoint as proxy ground truth
        gt = (preds.mean(axis=0) > 0.5).astype(int)
        selected_indices[w] = selector.select(competence[w], preds, gt)

    # --- save outputs ---------------------------------------------------
    np.save(out_dir / "selected_indices.npy", selected_indices)
    np.save(out_dir / "competence.npy", competence)
    np.save(out_dir / "drift_flags.npy", drift_flags)
    pd.DataFrame(selected_indices).to_csv(out_dir / "selected_indices.csv", index=False)

    meta = {
        "n_windows": int(n_windows),
        "window_size": window_size,
        "stride": stride,
        "k": k,
        "alpha": alpha,
        "selector": selector_name,
        "drift_method": drift_method,
        "drift_events": int(drift_detector.n_detections),
    }
    with open(out_dir / "selection_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"Selection complete. Outputs written to {out_dir}")


if __name__ == "__main__":
    main()
