# ABOUTME: Ablation runner orchestrates variant evaluation for component attribution.
# ABOUTME: Loads pre-computed blended scores, applies variant transforms, computes metrics.

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import hydra
import mlflow
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from cdade.data.dataset_paths import get_dataset_artifact_paths
from cdade.evaluation.metrics import compute_all_metrics

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_INJECTED_DIR = _PROJECT_ROOT / "data" / "injected"
_RESULTS_DIR = _PROJECT_ROOT / "results"


def load_variant_ground_truth(cfg: DictConfig, injected_dir: Path) -> np.ndarray:
    """Load binary anomaly mask and aggregate to time-series labels.

    Args:
        injected_dir: Path to injected data directory.

    Returns:
        1D numpy array of shape [n_timesteps] with binary labels.

    Raises:
        FileNotFoundError: If mask file not found.
    """
    artifact_paths = get_dataset_artifact_paths(cfg, project_root=_PROJECT_ROOT)
    mask_path = artifact_paths["mask"]

    if not mask_path.exists():
        raise FileNotFoundError(
            f"Ground truth mask not found at {mask_path}. "
            "Run `just data` or `uv run dvc repro inject` first."
        )

    mask_df = pd.read_parquet(mask_path)
    y_true = mask_df.values.astype(int).max(axis=1)
    return y_true


def load_variant_blended_scores(blended_path: Path, n_expected: int) -> np.ndarray:
    """Load blended detector scores from CSV.

    Args:
        blended_path: Path to blended_scores.csv.
        n_expected: Expected number of rows for validation.

    Returns:
        2D numpy array of shape [n_timesteps, n_detectors].

    Raises:
        FileNotFoundError: If file not found.
        ValueError: If shape does not match expected length.
    """
    if not blended_path.exists():
        raise FileNotFoundError(f"Blended scores not found at {blended_path}")

    df = pd.read_csv(blended_path)

    if len(df) != n_expected:
        raise ValueError(
            f"Blended scores shape mismatch: expected {n_expected} rows, got {len(df)}"
        )

    # Convert to numpy, dropping any index column
    scores = df.iloc[:, :].values.astype(np.float32)
    return scores


def apply_variant_transformation(blended_scores: np.ndarray, variant: str, cfg: Any) -> np.ndarray:
    """Apply variant-specific transformation to blended scores.

    For "full" and "no_reconciliation": return scores unchanged (identity).
    For "no_dynamic_selection": select top-k detectors by max score across time.
    For "no_diversity": return scores unchanged (selector ignores diversity internally).

    Args:
        blended_scores: Shape [n_timesteps, n_detectors].
        variant: One of "full", "no_reconciliation", "no_dynamic_selection", "no_diversity".
        cfg: Hydra configuration.

    Returns:
        Transformed scores array.
    """
    if variant in ("full", "no_reconciliation", "no_diversity"):
        # Identity: return unchanged
        return blended_scores

    if variant == "no_dynamic_selection":
        # Select top-k detectors by competence (mean score across time)
        k = cfg.selection.k if hasattr(cfg.selection, "k") else 5
        mean_scores = blended_scores.mean(axis=0)
        top_k_indices = np.argsort(mean_scores)[::-1][:k]
        return blended_scores[:, top_k_indices]

    return blended_scores


def compute_variant_metrics(
    y_true: np.ndarray,
    variant_scores: np.ndarray,
    nab_window: int = 4,
) -> dict[str, float]:
    """Compute metrics for a variant.

    Args:
        y_true: Ground truth labels.
        variant_scores: Variant-transformed anomaly scores.
        nab_window: NAB detection window.

    Returns:
        Dict with keys: auc_pr, nab, precision, recall, f1, threshold.
    """
    # If variant_scores is 2D, aggregate by max
    if variant_scores.ndim > 1:
        variant_scores = variant_scores.max(axis=1)

    metrics = compute_all_metrics(y_true, variant_scores, nab_window=nab_window)
    return metrics


def run_ablation_variants(
    cfg: Any,
    dataset_name: str,
    y_true: np.ndarray,
    blended_scores: np.ndarray,
    results_dir: Path,
) -> dict[str, dict]:
    """Run ablation study over all variants for one dataset.

    Args:
        cfg: Config-like object with cfg.selection.k and cfg.selection.alpha.
        dataset_name: Name of the dataset being processed.
        y_true: Ground truth labels for the full dataset.
        blended_scores: Shape [n_timesteps, n_detectors] pre-loaded scores.
        results_dir: Root results directory.

    Returns:
        Dict mapping variant name to metrics dict.
    """
    test_frac = getattr(getattr(cfg, "evaluation", None), "test_frac", 0.2)
    nab_window = getattr(getattr(cfg, "evaluation", None), "nab_window", 4)
    n_test = int(len(y_true) * test_frac)
    y_test = y_true[-n_test:]

    ablation_dir = results_dir / "ablation" / dataset_name
    ablation_dir.mkdir(parents=True, exist_ok=True)

    variants = ["full", "no_reconciliation", "no_dynamic_selection", "no_diversity"]
    all_metrics = {}

    for variant in variants:
        logger.info("[%s] Processing variant: %s", dataset_name, variant)
        variant_scores = apply_variant_transformation(blended_scores, variant, cfg)
        # Use test portion only
        variant_test = variant_scores[-n_test:]
        metrics = compute_variant_metrics(y_test, variant_test, nab_window=nab_window)
        all_metrics[variant] = metrics

        variant_file = ablation_dir / f"{variant}_metrics.json"
        with open(variant_file, "w", encoding="utf-8") as fh:
            json.dump(metrics, fh, indent=2)
        logger.info("[%s] Metrics written to %s", dataset_name, variant_file)

    return all_metrics


def _log_variant_to_mlflow(variant: str, metrics: dict[str, float]) -> None:
    """Log one variant's metrics to MLflow as a nested run.

    Args:
        variant: Variant name (used in run name, dataset_variant format).
        metrics: Metrics dict.
    """
    with mlflow.start_run(run_name=f"ablation_{variant}", nested=True):
        for key, value in metrics.items():
            mlflow.log_metric(key, value)


@hydra.main(
    config_path=str(_PROJECT_ROOT / "configs"),
    config_name="config",
    version_base=None,
)
def main(cfg: DictConfig | None = None) -> None:
    """Orchestrate ablation study.

    Args:
        cfg: Hydra configuration from configs/config.yaml. When invoked
            as a normal function (e.g. for static analysis) this may be None.
    """
    if cfg is None:
        raise ValueError("Hydra configuration is required")

    from cdade.data.dataset_paths import _iter_datasets

    mlflow_uri = cfg.experiment.mlflow_tracking_uri
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment("cdade_ablation")

    logger.info("Starting ablation study")
    combined_metrics: dict[str, dict[str, dict]] = {}

    with mlflow.start_run(run_name="ablation_study"):
        for dataset_name, artifact_paths in _iter_datasets(cfg, project_root=_PROJECT_ROOT):
            mask_path = artifact_paths["mask"]
            if not mask_path.exists():
                raise FileNotFoundError(f"Ground truth mask not found at {mask_path}.")
            mask_df = pd.read_parquet(mask_path)
            y_true = mask_df.values.astype(int).max(axis=1)

            blended_path = _RESULTS_DIR / "selection" / dataset_name / "blended_scores.csv"
            blended_scores = load_variant_blended_scores(blended_path, len(y_true))

            dataset_metrics = run_ablation_variants(
                cfg, dataset_name, y_true, blended_scores, _RESULTS_DIR
            )
            combined_metrics[dataset_name] = dataset_metrics

            for variant, metrics in dataset_metrics.items():
                _log_variant_to_mlflow(f"{dataset_name}_{variant}", metrics)

    # Write combined summary CSV
    rows = []
    for ds, variants in combined_metrics.items():
        for variant, metrics in variants.items():
            row = {"dataset": ds, "variant": variant}
            row.update(metrics)
            rows.append(row)
    summary_df = pd.DataFrame(rows)
    summary_path = _RESULTS_DIR / "ablation" / "summary.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info("Summary written to %s", summary_path)

    logger.info("Ablation complete")
    for ds, variants in combined_metrics.items():
        print(f"\n=== {ds} Ablation ===")
        for variant, metrics in variants.items():
            logger.info(
                "[%s] %s AUC-PR: %.4f NAB: %.4f", ds, variant, metrics["auc_pr"], metrics["nab"]
            )


if __name__ == "__main__":
    main()
