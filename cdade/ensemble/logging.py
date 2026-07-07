"""MLflow logging utilities for ensemble experiments.

Functions
--------
- log_experiment: Log parameters, metrics, and artifacts to MLflow run
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import mlflow
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def log_experiment(
    cfg: DictConfig,
    metrics: dict[str, Any],
    confusion_matrices: dict[str, Any] | None = None,
    competence_curves: dict[str, Any] | None = None,
    selection_history: dict[str, Any] | None = None,
) -> None:
    """Log experiment results to MLflow.

    Args:
        cfg: Hydra config object (will be logged as JSON)
        metrics: Dict of metric names to values
        confusion_matrices: Optional dict of confusion matrices per detector
        competence_curves: Optional dict of competence curves per detector
        selection_history: Optional dict of selection statistics

    Raises:
        RuntimeError: If MLflow tracking is not active
    """
    try:
        # Log parameters
        log_params(cfg)

        # Log metrics
        log_metrics(metrics)

        # Log artifacts
        if confusion_matrices:
            log_artifact("confusion_matrices.json", confusion_matrices)

        if competence_curves:
            log_artifact("competence_curves.json", competence_curves)

        if selection_history:
            log_artifact("selection_history.json", selection_history)

        active_run = mlflow.active_run()
        if active_run is not None:
            logger.info(f"Logged MLflow run: {active_run.info.run_id}")
    except Exception as e:
        logger.warning(f"Failed to log to MLflow: {e}")
        # Don't raise - MLflow logging is best-effort


def log_params(cfg: DictConfig) -> None:
    """Log config parameters to MLflow.

    Args:
        cfg: Hydra config object
    """
    try:
        # Convert to dict and log as JSON
        params_dict = dict(cfg)
        mlflow.log_params(params_dict)
        logger.debug(f"Logged {len(params_dict)} parameters")
    except Exception as e:
        logger.warning(f"Failed to log parameters: {e}")


def log_metrics(metrics: dict[str, Any]) -> None:
    """Log metrics to MLflow.

    Args:
        metrics: Dict of metric names to values
    """
    try:
        for name, value in metrics.items():
            mlflow.log_metric(name, value)
        logger.debug(f"Logged {len(metrics)} metrics")
    except Exception as e:
        logger.warning(f"Failed to log metrics: {e}")


def log_artifact(file_name: str, data: Any) -> None:
    """Write data to a file and log as MLflow artifact.

    Args:
        file_name: Name of the artifact file
        data: Data to write (dict, list, or JSON-serializable)
    """
    try:
        # Write to temp file
        temp_path = Path(file_name)
        with open(temp_path, "w") as f:
            json.dump(data, f, indent=2)

        # Log artifact
        mlflow.log_artifact(str(temp_path))

        # Cleanup
        temp_path.unlink()
        logger.debug(f"Logged artifact: {file_name}")
    except Exception as e:
        logger.warning(f"Failed to log artifact {file_name}: {e}")
