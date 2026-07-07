# ABOUTME: CLI entry point for the selection pipeline stage (DVC: select).
# ABOUTME: Writes per-dataset dynamic ensemble selection results to results/selection/{dataset}/.

"""CLI entry point for the selection pipeline stage.

Reads reconciled anomaly scores from results/reconciliation/{dataset}/,
runs per-window META-DES selection with optional drift detection,
and writes the active-subset indices + blended scores to results/selection/{dataset}/.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _run_single_dataset(
    cfg: DictConfig, dataset_name: str, recon_dir: Path, out_dir: Path
) -> dict[str, Any]:
    """Process selection for a single dataset.

    Args:
        cfg: Hydra configuration object.
        dataset_name: Name of the dataset being processed.
        recon_dir: Path to reconciliation results.
        out_dir: Path to write selection results.

    Returns:
        Dictionary containing selection results.
    """
    from cdade.registry import get_selector
    from cdade.selection import (
        DriftDetector,
        majority_vote_pseudo_labels,
        meta_des_competence,
    )

    # --- load reconciled scores ------------------------------------------
    scores_path = recon_dir / "leaf_forecasts_reconciled.csv"
    if not scores_path.exists():
        raise FileNotFoundError(f"Reconciled scores not found: {scores_path}")

    scores_df = pd.read_csv(scores_path, index_col=0)
    scores = scores_df.values.astype(np.float64)
    if scores.ndim == 1:
        scores = scores[:, np.newaxis]
    n_timepoints, n_series = scores.shape
    logger.info("[%s] Loaded scores: %s", dataset_name, scores.shape)

    # --- sliding-window setup -------------------------------------------
    window_size: int = cfg.selection.window
    stride: int = getattr(cfg.selection, "stride", 1)
    alpha: float = cfg.selection.alpha
    k: int = getattr(cfg.selection, "k", 5)
    threshold: float = getattr(cfg.selection, "threshold", 0.5)

    windows = np.arange(0, n_timepoints - window_size + 1, stride)
    n_windows = len(windows)

    if n_series == 0:
        logger.info("[%s] No detectors available; writing fallback outputs", dataset_name)
        result = _write_fallback_outputs(
            out_dir, n_windows, n_timepoints, cfg, window_size, stride, k, alpha
        )
        return result

    n_detectors = n_series
    windowed_scores = np.stack(
        [scores[s : s + window_size, :].T for s in windows],
        axis=0,
    )

    # --- pseudo-labels --------------------------------------------------
    score_summary = windowed_scores.mean(axis=2)[:, :, np.newaxis]
    pseudo_labels = majority_vote_pseudo_labels(score_summary, threshold=threshold)
    pseudo_labels_flat = pseudo_labels[:, :, 0]

    # --- competence -------------------------------------------------------
    pool_vote = (pseudo_labels_flat.mean(axis=1, keepdims=True) > 0.5).astype(np.int8)
    pool_vote_broadcast = np.tile(pool_vote, (1, n_detectors))

    competence = meta_des_competence(
        pseudo_labels_flat[:, :, np.newaxis].astype(np.int8),
        pool_vote_broadcast[:, :, np.newaxis].astype(np.int8),
    )[:, :, 0]

    # --- drift detection -----------------------------------------------
    drift_method: str = getattr(cfg.selection, "drift_method", "adwin")
    drift_detector = DriftDetector(method=drift_method)
    drift_flags = np.zeros(n_windows, dtype=bool)
    for w in range(n_windows):
        mean_comp = float(competence[w].mean())
        drift_flags[w] = drift_detector.update(mean_comp)
        if drift_flags[w]:
            competence[w] = 0.5

    # --- selector -------------------------------------------------------
    selector_name: str = cfg.selection.name
    selector = get_selector(selector_name)(k=k, alpha=alpha)

    selected_indices = np.zeros((n_windows, k), dtype=int)
    for w in range(n_windows):
        preds = (windowed_scores[w] > threshold).astype(int)
        gt = (preds.mean(axis=0) > 0.5).astype(int)
        selected = selector.select(competence[w], preds, gt)
        if len(selected) > 0:
            selected_indices[w] = selected

    # --- blended scores -------------------------------------------------
    blended = np.zeros(n_timepoints, dtype=float)
    for t in range(n_timepoints):
        w_idx = min(t // stride, n_windows - 1)
        sel = selected_indices[w_idx]
        if len(sel) > 0:
            blended[t] = scores[t, sel].mean()
        else:
            blended[t] = scores[t].mean()

    # --- save outputs ---------------------------------------------------
    np.save(out_dir / "selected_indices.npy", selected_indices)
    np.save(out_dir / "competence.npy", competence)
    np.save(out_dir / "drift_flags.npy", drift_flags)
    pd.DataFrame(selected_indices).to_csv(out_dir / "selected_indices.csv", index=False)

    blended_df = pd.DataFrame(
        np.tile(blended[:, np.newaxis], (1, n_detectors)),
        columns=[f"detector_{i}" for i in range(n_detectors)],
    )
    blended_df.to_csv(out_dir / "blended_scores.csv", index=False)
    pd.DataFrame(competence).to_csv(out_dir / "competence.csv", index=False)

    with open(out_dir / "active_detectors.json", "w") as f:
        json.dump(list(set(selected_indices.flatten().tolist())), f)
    with open(out_dir / "drift_history.json", "w") as f:
        json.dump(
            {
                "drift_flags": drift_flags.tolist(),
                "drift_count": int(drift_detector.n_detections),
            },
            f,
        )

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

    logger.info("[%s] Selection complete. Outputs written to %s", dataset_name, out_dir)

    return {
        "active_detectors": list(set(selected_indices.flatten().tolist())),
        "blended_scores": blended,
        "competence": competence,
        "drift_history": {
            "drift_flags": drift_flags.tolist(),
            "drift_count": int(drift_detector.n_detections),
        },
    }


def _write_fallback_outputs(
    out_dir: Path,
    n_windows: int,
    n_timepoints: int,
    cfg: DictConfig,
    window_size: int,
    stride: int,
    k: int,
    alpha: float,
) -> dict[str, Any]:
    """Write fallback outputs when no detectors are available.

    Args:
        out_dir: Output directory.
        n_windows: Number of windows.
        n_timepoints: Number of timepoints.
        cfg: Hydra configuration.
        window_size: Window size.
        stride: Window stride.
        k: Number of detectors to select.
        alpha: Diversity weight.

    Returns:
        Dictionary with fallback results.
    """
    selected_indices = np.zeros((n_windows, 0), dtype=int)
    competence = np.zeros((n_windows, 0), dtype=float)
    drift_flags = np.zeros(n_windows, dtype=bool)
    fallback_scores = np.zeros((n_timepoints, 1), dtype=float)

    np.save(out_dir / "selected_indices.npy", selected_indices)
    np.save(out_dir / "competence.npy", competence)
    np.save(out_dir / "drift_flags.npy", drift_flags)
    pd.DataFrame(fallback_scores, columns=["detector_0"]).to_csv(
        out_dir / "blended_scores.csv", index=False
    )
    pd.DataFrame(competence).to_csv(out_dir / "competence.csv", index=False)
    with open(out_dir / "active_detectors.json", "w") as f:
        json.dump([], f)
    with open(out_dir / "drift_history.json", "w") as f:
        json.dump({"drift_flags": [], "drift_count": 0}, f)

    meta = {
        "n_windows": int(n_windows),
        "window_size": window_size,
        "stride": stride,
        "k": k,
        "alpha": alpha,
        "selector": cfg.selection.name,
        "drift_method": getattr(cfg.selection, "drift_method", "adwin"),
        "drift_events": 0,
    }
    with open(out_dir / "selection_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return {
        "active_detectors": [],
        "blended_scores": fallback_scores,
        "competence": competence,
        "drift_history": {"drift_flags": [], "drift_count": 0},
    }


@hydra.main(config_path=str(_PROJECT_ROOT / "configs"), config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Run dynamic ensemble selection over reconciled detector scores — loops over datasets."""
    from cdade.data.dataset_paths import _iter_datasets

    for dataset_name, _paths in _iter_datasets(cfg, project_root=_PROJECT_ROOT):
        recon_dir = _PROJECT_ROOT / "results" / "reconciliation" / dataset_name
        out_dir = _PROJECT_ROOT / "results" / "selection" / dataset_name
        out_dir.mkdir(parents=True, exist_ok=True)
        _run_single_dataset(cfg, dataset_name, recon_dir, out_dir)


def run_select(cfg: DictConfig, dataset_name: str | None = None) -> dict[str, Any]:
    """Run dynamic ensemble selection over reconciled detector scores.

    Args:
        cfg: Hydra configuration object.
        dataset_name: Dataset name override. If None, inferred from cfg.

    Returns:
        Dictionary containing selection results.
    """
    if dataset_name is None:
        from cdade.data.dataset_paths import _dataset_name as _dn

        dataset_name = _dn(cfg)

    recon_dir = _PROJECT_ROOT / "results" / "reconciliation" / dataset_name
    out_dir = _PROJECT_ROOT / "results" / "selection" / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    return _run_single_dataset(cfg, dataset_name, recon_dir, out_dir)


if __name__ == "__main__":
    main()
