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

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


@hydra.main(config_path=str(_PROJECT_ROOT / "configs"), config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Run dynamic ensemble selection over reconciled detector scores — loops over datasets."""
    from cdade.data.dataset_paths import _iter_datasets
    from cdade.registry import get_selector
    from cdade.selection import (
        DriftDetector,
        majority_vote_pseudo_labels,
        meta_des_competence,
    )

    for dataset_name, _paths in _iter_datasets(cfg, project_root=_PROJECT_ROOT):
        recon_dir = _PROJECT_ROOT / "results" / "reconciliation" / dataset_name
        out_dir = _PROJECT_ROOT / "results" / "selection" / dataset_name
        out_dir.mkdir(parents=True, exist_ok=True)

        # --- load reconciled scores ------------------------------------------
        scores_path = recon_dir / "leaf_forecasts_reconciled.csv"
        if not scores_path.exists():
            raise FileNotFoundError(f"Reconciled scores not found: {scores_path}")

        scores_df = pd.read_csv(scores_path, index_col=0)
        # Shape expected after reconciliation: (n_timepoints, n_series)
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
            continue

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

        # --- selector -------------------------------------------------------
        selector_name: str = cfg.selection.name
        selector = get_selector(selector_name)(k=k, alpha=alpha)

        selected_indices = np.zeros((n_windows, k), dtype=int)
        for w in range(n_windows):
            # predictions: (n_detectors, window_size)
            preds = (windowed_scores[w] > threshold).astype(int)
            # majority label per timepoint as proxy ground truth
            gt = (preds.mean(axis=0) > 0.5).astype(int)
            selected = selector.select(competence[w], preds, gt)
            if len(selected) == 0:
                selected_indices[w, :0] = selected
            else:
                selected_indices[w] = selected

        # Build blended scores: aggregate across selected detectors per timepoint
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


def run_select(cfg: DictConfig) -> dict:
    """Run dynamic ensemble selection over reconciled detector scores.

    Args:
        cfg: Hydra configuration object

    Returns:
        Dictionary containing selection results
    """
    from cdade.registry import get_selector
    from cdade.selection import (
        DriftDetector,
        ensemble_q_diversity,
        majority_vote_pseudo_labels,
        meta_des_competence,
    )

    recon_dir = _PROJECT_ROOT / "results" / "reconciliation"
    out_dir = _PROJECT_ROOT / "results" / "selection"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- load reconciled scores ------------------------------------------
    scores_path = recon_dir / "leaf_forecasts_reconciled.csv"
    if not scores_path.exists():
        raise FileNotFoundError(f"Reconciled scores not found: {scores_path}")

    scores_df = pd.read_csv(scores_path, index_col=0)
    # Shape expected after reconciliation: (n_timepoints, n_series)
    scores = scores_df.values.astype(np.float64)
    if scores.ndim == 1:
        scores = scores[:, np.newaxis]
    n_timepoints, n_series = scores.shape
    logger.info(f"Loaded scores: {scores.shape}")

    # --- sliding-window setup -------------------------------------------
    window_size: int = cfg.selection.window
    stride: int = getattr(cfg.selection, "stride", 1)
    alpha: float = cfg.selection.alpha

    # --- pseudo-label generation ---------------------------------------
    if scores.shape[1] == 0:
        n_windows = len(np.arange(0, n_timepoints - window_size + 1, stride))
        competence = np.zeros((n_windows, 0), dtype=float)
        selected_indices = np.array([], dtype=int)
        blended_scores = np.zeros((n_timepoints, 1), dtype=float)
        drift_history = {"drift_flags": [], "drift_count": 0}
        with open(out_dir / "active_detectors.json", "w") as f:
            json.dump([], f)
        pd.DataFrame(blended_scores, columns=["detector_0"]).to_csv(
            out_dir / "blended_scores.csv", index=False
        )
        pd.DataFrame(competence).to_csv(out_dir / "competence.csv", index=False)
        with open(out_dir / "drift_history.json", "w") as f:
            json.dump(drift_history, f)
        logger.info("Selection complete: no detectors available")
        return {
            "active_detectors": [],
            "blended_scores": blended_scores,
            "competence": {},
            "drift_history": drift_history,
        }

    pseudo_labels = majority_vote_pseudo_labels(scores[:, :, np.newaxis], threshold=0.5)

    # --- competence estimation -------------------------------------------
    competence = meta_des_competence(
        pseudo_labels,
        pseudo_labels,
    )[:, :, 0]

    # --- diversity calculation -------------------------------------------
    diversity = ensemble_q_diversity(
        scores,
        window_size,
    )

    # --- subset selection -----------------------------------------------
    selector = get_selector(cfg.selection.name)
    selected_indices, blended_scores = selector.fit_predict(
        competence=competence,
        diversity=diversity,
        k=cfg.selection.k,
        alpha=alpha,
    )

    # --- drift detection -----------------------------------------------
    detector = DriftDetector(method=cfg.selection.drift_method)
    _ = detector.fit_predict(competence)

    # --- save results ----------------------------------------------------
    # Save selected detectors
    selected_detectors = set(selected_indices)
    with open(out_dir / "active_detectors.json", "w") as f:
        json.dump(list(selected_detectors), f)

    # Save blended scores
    blended_scores_df = pd.DataFrame(
        blended_scores,
        columns=[f"detector_{i}" for i in range(n_series)],
    )
    blended_scores_df.to_csv(out_dir / "blended_scores.csv", index=False)

    # Save competence
    competence_df = pd.DataFrame(competence)
    competence_df.to_csv(out_dir / "competence.csv", index=False)

    # Save drift history
    drift_history = {
        "drift_flags": detector.drift_detected.tolist(),
        "drift_count": detector._drift_count,
    }
    with open(out_dir / "drift_history.json", "w") as f:
        json.dump(drift_history, f)

    logger.info(f"Selection complete: {len(selected_detectors)} detectors selected")

    return {
        "active_detectors": selected_detectors,
        "blended_scores": blended_scores,
        "competence": competence_df.to_dict(),
        "drift_history": drift_history,
    }


if __name__ == "__main__":
    main()
