"""DVC entry-point for the baselines stage.

Runs B1–B5 baselines on the same injected data used by the CDADE pipeline,
logs per-baseline params/metrics to MLflow, and writes score artefacts to
results/baselines/.

Designed to be invoked by DVC as:
    uv run python -m cdade.baselines.run_baselines

Uses @hydra.main so the module is safely importable without triggering
Hydra initialisation at import time.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import hydra
import mlflow
import numpy as np
import pandas as pd
from omegaconf import DictConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_INJECTED_DIR = _PROJECT_ROOT / "data" / "injected"
_RESULTS_DIR = _PROJECT_ROOT / "results" / "baselines"


def _load_injected_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load injected counts and anomaly mask for SIVEP.

    Returns:
        Tuple of (counts DataFrame, mask DataFrame).

    Raises:
        FileNotFoundError: If DVC injected data has not been produced yet.
    """
    counts_path = _INJECTED_DIR / "sivep_counts_injected.parquet"
    mask_path = _INJECTED_DIR / "sivep_counts_mask.parquet"

    if not counts_path.exists():
        raise FileNotFoundError(
            f"Injected data not found at {counts_path}. "
            "Run `just data` or `uv run dvc repro inject` first."
        )

    counts = pd.read_parquet(counts_path)
    mask = pd.read_parquet(mask_path)
    return counts, mask


def _train_test_split(
    counts: pd.DataFrame, mask: pd.DataFrame, val_frac: float = 0.6, test_frac: float = 0.2
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split data into train / validation / test arrays.

    Args:
        counts: Raw injected counts (time × series).
        mask: Binary anomaly mask (time × series), 1 = anomaly.
        val_frac: Fraction of data used for training.
        test_frac: Fraction used for test; remainder is validation.

    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test).
    """
    X = counts.values.astype(float)
    y = mask.values.astype(int).max(axis=1)  # row-wise: anomaly if any series flagged

    n = len(X)
    n_train = int(n * val_frac)
    n_val = int(n * test_frac)

    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train : n_train + n_val], y[n_train : n_train + n_val]
    X_test, y_test = X[n_train + n_val :], y[n_train + n_val :]

    return X_train, X_val, X_test, y_train, y_val, y_test


def _log_baseline(name: str, scores: np.ndarray, labels: np.ndarray, params: dict) -> dict:
    """Log one baseline run to MLflow and return a metrics summary.

    Args:
        name: Baseline identifier (e.g. "b3_ensemble_average").
        scores: Anomaly scores (shape: [n_test]).
        labels: Ground-truth binary labels (shape: [n_test]).
        params: Extra params to log alongside the baseline name.

    Returns:
        Dict of computed metrics.
    """
    from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score

    threshold = float(np.median(scores))
    preds = (scores >= threshold).astype(int)

    metrics = {
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "auc_pr": float(average_precision_score(labels, scores)),
        "threshold": threshold,
    }

    with mlflow.start_run(run_name=name, nested=True):
        mlflow.log_param("baseline", name)
        for k, v in params.items():
            mlflow.log_param(k, v)
        for k, v in metrics.items():
            mlflow.log_metric(k, v)

    logger.info(
        "%s — F1=%.3f AUC-PR=%.3f",
        name,
        metrics["f1"],
        metrics["auc_pr"],
    )
    return metrics


# ---------------------------------------------------------------------------
# Per-baseline runners
# ---------------------------------------------------------------------------


def _run_b1(counts: pd.DataFrame, cfg: DictConfig) -> tuple[np.ndarray, np.ndarray, dict]:
    """Run B1: Farrington/Noufaily on the first leaf series."""
    from cdade.baselines.farrington import FarringtonConfig, FarringtonDetector

    series = counts.iloc[:, 0].values
    n = len(series)
    n_train = int(n * 0.6)

    train_series = series[:n_train]
    test_series = series[n_train:]

    config = FarringtonConfig(
        seasonal_period=52,
        z_threshold=3.0,
        llr_threshold=0.05,
        min_obs=12,
    )
    detector = FarringtonDetector(config)
    detector.fit(train_series)

    scores = detector.score(test_series)
    params = {"method": "farrington", "z_threshold": config.z_threshold}
    return scores, params


def _run_b2(
    X_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, X_test: np.ndarray, cfg: DictConfig
) -> tuple[np.ndarray, dict]:
    """Run B2: best single detector selected on validation set."""
    import cdade.detectors.cblof  # noqa: F401
    import cdade.detectors.hbos  # noqa: F401
    import cdade.detectors.iforest  # noqa: F401
    import cdade.detectors.lof  # noqa: F401
    import cdade.detectors.pca  # noqa: F401
    from cdade.baselines.single_best import BestSingleDetector

    # fit uses internal train/val split; we pass full train here
    combined = np.concatenate([X_train, X_val], axis=0)
    labels_combined = np.concatenate(
        [np.zeros(len(X_train), dtype=int), np.ones(len(X_val), dtype=int)]
    )
    detector = BestSingleDetector()
    detector.fit(combined, labels_combined)

    scores = detector.score(X_test)
    params = {"method": "best_single", "selected": detector.best_detector_name}
    return scores, params


def _run_b3(X_train: np.ndarray, X_test: np.ndarray, cfg: DictConfig) -> tuple[np.ndarray, dict]:
    """Run B3: full ensemble average (AOM)."""
    import cdade.detectors.hbos  # noqa: F401
    import cdade.detectors.iforest  # noqa: F401
    import cdade.detectors.lof  # noqa: F401
    import cdade.detectors.pca  # noqa: F401
    from cdade.baselines.ensemble_average import EnsembleAverageConfig, EnsembleAverageDetector

    detector = EnsembleAverageDetector(EnsembleAverageConfig(normalize=True))
    detector.fit(X_train)
    scores = detector.score(X_test)
    params = {"method": "ensemble_average", "n_detectors": len(detector.fitted_detectors)}
    return scores, params


def _run_b4(
    X_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    cfg: DictConfig,
) -> tuple[np.ndarray, dict]:
    """Run B4: static top-k greedy set-cover."""
    import cdade.detectors.hbos  # noqa: F401
    import cdade.detectors.iforest  # noqa: F401
    import cdade.detectors.lof  # noqa: F401
    import cdade.detectors.pca  # noqa: F401
    from cdade.baselines.static_topk import StaticTopKConfig, StaticTopKDetector

    detector = StaticTopKDetector(StaticTopKConfig(k=0, alpha=0.5))
    detector.fit(X_train, X_val, y_val)
    scores = detector.score(X_test)
    params = {
        "method": "static_topk",
        "k": len(detector.selected_indices),
        "alpha": 0.5,
    }
    return scores, params


def _run_b5(X_train: np.ndarray, X_test: np.ndarray, cfg: DictConfig) -> tuple[np.ndarray, dict]:
    """Run B5: reconciliation + EVT, no dynamic selection."""
    import cdade.detectors.hbos  # noqa: F401
    import cdade.detectors.iforest  # noqa: F401
    import cdade.detectors.lof  # noqa: F401
    import cdade.detectors.pca  # noqa: F401
    from cdade.baselines.reconciliation_evt import (
        ReconciliationEVTConfig,
        ReconciliationEVTDetector,
    )

    detector = ReconciliationEVTDetector(
        ReconciliationEVTConfig(contamination=0.05, alpha_evt=0.05)
    )
    detector.fit(X_train)
    scores = detector.score(X_test)
    params = {"method": "reconciliation_evt", "contamination": 0.05, "alpha_evt": 0.05}
    return scores, params


# ---------------------------------------------------------------------------
# Main entry-point
# ---------------------------------------------------------------------------


@hydra.main(
    config_path=str(_PROJECT_ROOT / "configs"),
    config_name="config",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    """Run all baselines and log results to MLflow.

    Args:
        cfg: Hydra configuration loaded from configs/config.yaml.
    """
    seed = int(cfg.experiment.seed)
    np.random.seed(seed)

    mlflow_uri = cfg.experiment.mlflow_tracking_uri
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment("cdade_baselines")

    logger.info("Loading injected data from %s", _INJECTED_DIR)
    counts, mask = _load_injected_data()

    X_train, X_val, X_test, y_train, y_val, y_test = _train_test_split(counts, mask)
    logger.info(
        "Split: train=%d val=%d test=%d anomaly_rate=%.2f%%",
        len(X_train),
        len(X_val),
        len(X_test),
        100 * y_test.mean(),
    )

    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_metrics: dict[str, dict] = {}

    with mlflow.start_run(run_name=f"baselines_seed{seed}"):
        mlflow.log_param("seed", seed)
        mlflow.log_param("n_train", len(X_train))
        mlflow.log_param("n_val", len(X_val))
        mlflow.log_param("n_test", len(X_test))

        # B1 — operates on univariate series; y_test for single series
        try:
            b1_scores, b1_params = _run_b1(counts, cfg)
            # Align length with test set (B1 is univariate)
            n_b1 = min(len(b1_scores), len(y_test))
            b1_metrics = _log_baseline("b1_farrington", b1_scores[:n_b1], y_test[:n_b1], b1_params)
            all_metrics["b1_farrington"] = b1_metrics
            np.save(_RESULTS_DIR / "b1_scores.npy", b1_scores)
        except Exception as exc:
            logger.warning("B1 failed: %s", exc)

        # B2
        try:
            b2_scores, b2_params = _run_b2(X_train, X_val, y_val, X_test, cfg)
            b2_metrics = _log_baseline("b2_best_single", b2_scores, y_test, b2_params)
            all_metrics["b2_best_single"] = b2_metrics
            np.save(_RESULTS_DIR / "b2_scores.npy", b2_scores)
        except Exception as exc:
            logger.warning("B2 failed: %s", exc)

        # B3
        try:
            b3_scores, b3_params = _run_b3(X_train, X_test, cfg)
            b3_metrics = _log_baseline("b3_ensemble_average", b3_scores, y_test, b3_params)
            all_metrics["b3_ensemble_average"] = b3_metrics
            np.save(_RESULTS_DIR / "b3_scores.npy", b3_scores)
        except Exception as exc:
            logger.warning("B3 failed: %s", exc)

        # B4
        try:
            b4_scores, b4_params = _run_b4(X_train, X_val, y_val, X_test, cfg)
            b4_metrics = _log_baseline("b4_static_topk", b4_scores, y_test, b4_params)
            all_metrics["b4_static_topk"] = b4_metrics
            np.save(_RESULTS_DIR / "b4_scores.npy", b4_scores)
        except Exception as exc:
            logger.warning("B4 failed: %s", exc)

        # B5
        try:
            b5_scores, b5_params = _run_b5(X_train, X_test, cfg)
            b5_metrics = _log_baseline("b5_reconciliation_evt", b5_scores, y_test, b5_params)
            all_metrics["b5_reconciliation_evt"] = b5_metrics
            np.save(_RESULTS_DIR / "b5_scores.npy", b5_scores)
        except Exception as exc:
            logger.warning("B5 failed: %s", exc)

        # Summary metric file consumed by DVC
        metrics_path = _PROJECT_ROOT / "results" / "baselines_metrics.json"
        with open(metrics_path, "w") as fh:
            json.dump(all_metrics, fh, indent=2)
        mlflow.log_artifact(str(metrics_path))

    logger.info("Baseline results written to %s", _RESULTS_DIR)
    print("\n=== Baselines Complete ===")
    for name, m in all_metrics.items():
        print(f"  {name}: F1={m.get('f1', 'N/A'):.3f}  AUC-PR={m.get('auc_pr', 'N/A'):.3f}")
    print("===\n")


if __name__ == "__main__":
    main()
