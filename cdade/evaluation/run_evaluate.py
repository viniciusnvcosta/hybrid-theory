# ABOUTME: DVC entry-point that orchestrates evaluation across CDADE and baselines.
# ABOUTME: Loads ground truth, scores, and outputs metrics.json and per-method CSVs.

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


def load_ground_truth(mask_path: Path) -> np.ndarray:
    """Load binary anomaly mask and aggregate to time-series labels.

    Args:
        mask_path: Path to injected mask parquet file.

    Returns:
        1D numpy array of shape [n_timesteps] with binary labels (max across series).

    Raises:
        FileNotFoundError: If mask file not found.
    """
    if not mask_path.exists():
        raise FileNotFoundError(
            f"Ground truth mask not found at {mask_path}. "
            "Run `just data` or `uv run dvc repro inject` first."
        )

    mask_df = pd.read_parquet(mask_path)
    y_true = mask_df.values.astype(int).max(axis=1)  # Row-wise max: anomaly if any series
    return y_true


def load_blended_scores(blended_path: Path, n_expected: int) -> pd.DataFrame:
    """Load blended detector scores from CSV.

    Args:
        blended_path: Path to blended_scores.csv.
        n_expected: Expected number of rows for validation.

    Returns:
        DataFrame with shape [n_timesteps, n_detectors].

    Raises:
        FileNotFoundError: If file not found.
        ValueError: If shape does not match expected length.
    """
    if not blended_path.exists():
        raise FileNotFoundError(f"Blended scores not found at {blended_path}")

    df = pd.read_csv(blended_path)

    if len(df) != n_expected:
        raise ValueError(
            f"Blended scores shape mismatch: expected {n_expected} rows, got {len(df)}. "
            f"Check if detector stage produced correct output."
        )

    return df


def load_baseline_scores(baselines_dir: Path) -> dict[str, np.ndarray]:
    """Load baseline scores b1–b5 from .npy files.

    Args:
        baselines_dir: Path to results/baselines directory.

    Returns:
        Dict mapping method names to score arrays.
    """
    baselines = {}

    for i in range(1, 6):
        baseline_path = baselines_dir / f"b{i}_scores.npy"
        if baseline_path.exists():
            baselines[f"b{i}"] = np.load(baseline_path)
        else:
            logger.warning(f"Baseline {i} scores not found at {baseline_path}")

    return baselines


def evaluate_all_methods(
    y_test: np.ndarray,
    cdade_scores: np.ndarray,
    baselines: dict[str, np.ndarray],
    nab_window: int = 4,
) -> dict[str, dict[str, float]]:
    """Compute metrics for CDADE and all loaded baselines.

    Args:
        y_test: Test set binary labels.
        cdade_scores: CDADE anomaly scores (same length as y_test).
        baselines: Dict of baseline name to score array.
        nab_window: NAB detection window half-width.

    Returns:
        Dict mapping method name to metrics dict (auc_pr, nab, precision, recall, f1, threshold).
    """
    all_metrics: dict[str, dict[str, float]] = {}

    # CDADE
    cdade_metrics = compute_all_metrics(y_test, cdade_scores, nab_window=nab_window)
    all_metrics["cdade"] = cdade_metrics

    # Baselines
    for method, scores in baselines.items():
        # Ensure same length; some baselines may be shorter due to univariate processing
        n_test = len(y_test)
        if len(scores) < n_test:
            logger.warning(f"{method} scores shorter than test set: {len(scores)} vs {n_test}")
            # Pad or truncate
            scores = np.pad(scores, (0, max(0, n_test - len(scores))), mode="edge")[:n_test]
        elif len(scores) > n_test:
            scores = scores[:n_test]

        metrics = compute_all_metrics(y_test, scores, nab_window=nab_window)
        all_metrics[method] = metrics

    return all_metrics


def save_metrics_json(all_metrics: dict[str, dict], output_path: Path) -> None:
    """Write metrics dict to JSON file.

    Args:
        all_metrics: Dict mapping method name to metrics.
        output_path: Where to write metrics.json.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as fh:
        json.dump(all_metrics, fh, indent=2)
    logger.info(f"Metrics written to {output_path}")


def save_per_method_csvs(all_metrics: dict[str, dict], eval_dir: Path) -> None:
    """Write per-method metrics to individual CSV files.

    Args:
        all_metrics: Dict mapping method name to metrics.
        eval_dir: Output directory for per-method CSVs.
    """
    eval_dir.mkdir(parents=True, exist_ok=True)

    for method, metrics in all_metrics.items():
        csv_path = eval_dir / f"{method}_metrics.csv"
        df = pd.DataFrame([metrics])  # One row per method
        df.to_csv(csv_path, index=False)
        logger.info(f"  {method}: {csv_path}")


def _log_method_to_mlflow(method: str, metrics: dict[str, float]) -> None:
    """Log one method's metrics to MLflow as a nested run.

    Args:
        method: Method name (cdade, b1, etc.).
        metrics: Metrics dict.
    """
    with mlflow.start_run(run_name=method, nested=True):
        mlflow.log_param("method", method)
        for key, value in metrics.items():
            mlflow.log_metric(key, value)


@hydra.main(
    config_path=str(_PROJECT_ROOT / "configs"),
    config_name="config",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    """Orchestrate evaluation of CDADE and baselines for all active datasets.

    Args:
        cfg: Hydra configuration from configs/config.yaml.
    """
    from cdade.data.dataset_paths import _iter_datasets

    mlflow_uri = cfg.experiment.mlflow_tracking_uri
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment("cdade_evaluation")

    test_frac = cfg.evaluation.test_frac
    nab_window = cfg.evaluation.nab_window

    # Use _PROJECT_ROOT dynamically to support mocking in tests
    results_dir = _PROJECT_ROOT / "results"

    with mlflow.start_run(run_name="evaluate"):
        mlflow.log_param("test_frac", test_frac)
        mlflow.log_param("nab_window", nab_window)

        for dataset_name, artifact_paths in _iter_datasets(cfg, project_root=_PROJECT_ROOT):
            logger.info("Evaluating dataset: %s", dataset_name)

            # Ground truth
            mask_path = artifact_paths["mask"]
            if not mask_path.exists():
                raise FileNotFoundError(
                    f"Ground truth mask not found at {mask_path}. "
                    "Run `just data` or `uv run dvc repro inject` first."
                )
            mask_df = pd.read_parquet(mask_path)
            y_true = mask_df.values.astype(int).max(axis=1)
            n_test = int(len(y_true) * test_frac)
            y_test = y_true[-n_test:]
            logger.info(
                "[%s] Test size: %d (anomaly rate: %.1f%%)",
                dataset_name,
                len(y_test),
                100 * y_test.mean(),
            )

            # CDADE blended scores
            blended_path = results_dir / "selection" / dataset_name / "blended_scores.csv"
            blended_df = load_blended_scores(blended_path, len(y_true))
            cdade_test = blended_df.values[-n_test:].max(axis=1)
            if len(cdade_test) != len(y_test):
                raise ValueError(
                    f"[{dataset_name}] CDADE scores/labels shape mismatch: "
                    f"{len(cdade_test)} vs {len(y_test)}."
                )

            # Baselines
            baselines_dir = results_dir / "baselines" / dataset_name
            baselines = load_baseline_scores(baselines_dir)

            # Metrics
            all_metrics = evaluate_all_methods(y_test, cdade_test, baselines, nab_window)

            # Per-dataset outputs
            metrics_out = results_dir / "metrics" / dataset_name / "metrics.json"
            save_metrics_json(all_metrics, metrics_out)

            eval_dir = results_dir / "evaluation" / dataset_name
            logger.info("[%s] Writing per-method CSV files to %s", dataset_name, eval_dir)
            save_per_method_csvs(all_metrics, eval_dir)

            # MLflow nested run per dataset
            with mlflow.start_run(run_name=dataset_name, nested=True):
                mlflow.log_param("dataset", dataset_name)
                mlflow.log_metric("n_test", len(y_test))
                mlflow.log_metric("anomaly_rate_test", float(y_test.mean()))
                for method, metrics in all_metrics.items():
                    for key, value in metrics.items():
                        mlflow.log_metric(f"{method}_{key}", value)
                mlflow.log_artifact(str(metrics_out))

            print(f"\n=== {dataset_name} Evaluation ===")
            for method, metrics in all_metrics.items():
                print(
                    f"  {method}: AUC-PR={metrics['auc_pr']:.3f} "
                    f"NAB={metrics['nab']:.3f}  F1={metrics['f1']:.3f}"
                )

    logger.info("Evaluation complete")


if __name__ == "__main__":
    main()
