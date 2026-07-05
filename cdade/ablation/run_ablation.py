# ABOUTME: Ablation runner orchestrates variant evaluation for component attribution.
# ABOUTME: Loads pre-computed blended scores, applies variant transforms, computes metrics.

from __future__ import annotations

import json
import logging
from pathlib import Path

import hydra
import mlflow
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from cdade.evaluation.metrics import compute_all_metrics

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_INJECTED_DIR = _PROJECT_ROOT / "data" / "injected"
_RESULTS_DIR = _PROJECT_ROOT / "results"


def load_variant_ground_truth(injected_dir: Path) -> np.ndarray:
    """Load binary anomaly mask and aggregate to time-series labels.

    Args:
        injected_dir: Path to injected data directory.

    Returns:
        1D numpy array of shape [n_timesteps] with binary labels.

    Raises:
        FileNotFoundError: If mask file not found.
    """
    mask_path = injected_dir / "sivep_counts_mask.parquet"

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


def apply_variant_transformation(
    blended_scores: np.ndarray, variant: str, cfg: DictConfig
) -> np.ndarray:
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


def run_ablation_variants(results_dir: Path | None = None) -> dict[str, dict]:
    """Run ablation study over all variants.

    Loads pre-computed blended scores from results/selection/blended_scores.csv,
    applies variant-specific transformations, and computes metrics for each.

    Args:
        results_dir: Results directory (default: _RESULTS_DIR).

    Returns:
        Dict mapping variant name to metrics dict.
    """
    if results_dir is None:
        results_dir = _RESULTS_DIR

    results_dir = Path(results_dir)
    ablation_dir = results_dir / "ablation"
    ablation_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading ground truth from {_INJECTED_DIR}")
    y_true = load_variant_ground_truth(_INJECTED_DIR)

    logger.info(f"Loading blended scores from {results_dir / 'selection'}")
    blended_path = results_dir / "selection" / "blended_scores.csv"
    blended_scores = load_variant_blended_scores(blended_path, len(y_true))

    # Create a minimal config object for variant transformations
    cfg = type("Config", (), {})()
    cfg.selection = type("Selection", (), {})()
    cfg.selection.k = 5
    cfg.selection.alpha = 0.5

    variants = ["full", "no_reconciliation", "no_dynamic_selection", "no_diversity"]
    all_metrics = {}

    for variant in variants:
        logger.info(f"Processing variant: {variant}")

        # Apply variant transformation
        variant_scores = apply_variant_transformation(blended_scores, variant, cfg)

        # Compute metrics
        metrics = compute_variant_metrics(y_true, variant_scores, nab_window=4)
        all_metrics[variant] = metrics

        # Write variant metrics to JSON
        variant_file = ablation_dir / f"{variant}_metrics.json"
        with open(variant_file, "w") as fh:
            json.dump(metrics, fh, indent=2)
        logger.info(f"  Metrics written to {variant_file}")

    # Write summary CSV
    summary_df = pd.DataFrame(all_metrics).T
    summary_df.insert(0, "variant", summary_df.index)
    summary_path = ablation_dir / "summary.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"Summary written to {summary_path}")

    return all_metrics


def _log_variant_to_mlflow(variant: str, metrics: dict[str, float]) -> None:
    """Log one variant's metrics to MLflow as a nested run.

    Args:
        variant: Variant name.
        metrics: Metrics dict.
    """
    with mlflow.start_run(run_name=f"ablation_{variant}", nested=True):
        mlflow.log_param("variant", variant)
        for key, value in metrics.items():
            mlflow.log_metric(key, value)


@hydra.main(
    config_path=str(_PROJECT_ROOT / "configs"),
    config_name="config",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    """Orchestrate ablation study.

    Args:
        cfg: Hydra configuration from configs/config.yaml.
    """
    mlflow_uri = cfg.experiment.mlflow_tracking_uri
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment("cdade_ablation")

    logger.info("Starting ablation study")
    all_metrics = run_ablation_variants(results_dir=_RESULTS_DIR)

    # Log to MLflow
    with mlflow.start_run(run_name="ablation_study"):
        for variant, metrics in all_metrics.items():
            _log_variant_to_mlflow(variant, metrics)

    logger.info("Ablation complete")
    print("\n=== Ablation Study Complete ===")
    for variant, metrics in all_metrics.items():
        auc_pr = metrics["auc_pr"]
        nab = metrics["nab"]
        print(f"{variant:25s} AUC-PR: {auc_pr:.4f}  NAB: {nab:.4f}")


if __name__ == "__main__":
    main()
