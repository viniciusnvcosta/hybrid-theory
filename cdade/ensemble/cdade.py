"""CDADE end-to-end orchestrator.

Wires L1 (detectors) → L2 (reconciliation) → L3 (selection) with
MLflow tracking and result aggregation.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from cdade.detectors.run_detect import run_detect
from cdade.ensemble.logging import log_experiment
from cdade.reconciliation.run_reconcile import run_reconcile
from cdade.selection.run_select import run_select

logger = logging.getLogger(__name__)


def _json_default(value: Any) -> Any:
    """Convert numpy and other non-JSON-native values to JSON-safe types."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    if isinstance(value, pd.Series):
        return value.to_dict()
    if isinstance(value, np.integer | np.floating):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): v for k, v in value.items()}
    if isinstance(value, list | tuple):
        return list(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


class CDADEOrchestrator:
    """Orchestrates the complete CDADE pipeline.

    Attributes:
        cfg: Hydra configuration
        results: Dictionary to store pipeline results
    """

    def __init__(self, cfg: DictConfig):
        """Initialize orchestrator.

        Args:
            cfg: Hydra configuration object
        """
        self.cfg = cfg
        self.results: dict[str, Any] = {}

    def run(self) -> dict[str, Any]:
        """Run the complete CDADE pipeline end-to-end.

        Flow:
            1. Load config
            2. Run detectors (L1)
            3. Reconcile (L2)
            4. Select (L3)
            5. Aggregate metrics
            6. Log to MLflow

        Returns:
            Dictionary containing all pipeline results
        """
        logger.info("Starting CDADE pipeline")

        # Step 1: Load config (already done in __init__)
        # Step 2: Run detectors (L1)
        logger.info("Running detectors (L1)...")
        self.results["detectors"] = run_detect(self.cfg)

        # Step 3: Reconcile (L2)
        logger.info("Running reconciliation (L2)...")
        self.results["reconciliation"] = run_reconcile(self.cfg)

        # Step 4: Select (L3)
        logger.info("Running selection (L3)...")
        self.results["selection"] = run_select(self.cfg)

        # Step 5: Aggregate metrics
        logger.info("Aggregating metrics...")
        metrics = self._aggregate_metrics()

        # Step 6: Log to MLflow
        logger.info("Logging to MLflow...")
        self._log_to_mlflow(metrics)

        logger.info("CDADE pipeline complete")
        output_dir = Path("results/ensemble")
        output_dir.mkdir(parents=True, exist_ok=True)
        summary = json.loads(json.dumps(self.results, default=_json_default))
        (output_dir / "ensemble_summary.json").write_text(json.dumps(summary, indent=2))
        metrics_path = Path("results/ensemble_metrics.json")
        metrics_path.write_text(json.dumps(self._aggregate_metrics(), indent=2))
        return self.results

    def _aggregate_metrics(self) -> dict[str, Any]:
        """Aggregate metrics from pipeline stages.

        Returns:
            Dictionary of aggregated metrics
        """
        metrics = {}

        # Detector metrics
        detectors_output = self.results.get("detectors", {})
        if "scores" in detectors_output:
            scores = detectors_output["scores"]
            metrics["detector_count"] = len(scores) if isinstance(scores, list) else 1

        # Reconciliation metrics
        reconciliation_output = self.results.get("reconciliation", {})
        if "coherent_scores" in reconciliation_output:
            coherent_scores = reconciliation_output["coherent_scores"]
            metrics["reconciled_count"] = (
                len(coherent_scores) if isinstance(coherent_scores, list) else 1
            )

        # Selection metrics
        selection_output = self.results.get("selection", {})
        if "active_detectors" in selection_output:
            active_detectors = selection_output["active_detectors"]
            metrics["selected_detectors"] = (
                len(active_detectors) if isinstance(active_detectors, list) else 1
            )

        # Competence statistics
        competence_data = selection_output.get("competence", {})
        if "windowed_competence" in competence_data:
            competence = competence_data["windowed_competence"]
            if isinstance(competence, dict):
                all_values = np.concatenate([v.flatten() for v in competence.values()])
                if len(all_values) > 0:
                    metrics["mean_competence"] = float(np.mean(all_values))
                    metrics["std_competence"] = float(np.std(all_values))
                    metrics["min_competence"] = float(np.min(all_values))
                    metrics["max_competence"] = float(np.max(all_values))

        return metrics

    def _log_to_mlflow(self, metrics: dict[str, Any]) -> None:
        """Log results to MLflow.

        Args:
            metrics: Aggregated metrics dictionary
        """
        try:
            with mlflow.start_run():
                log_experiment(
                    cfg=self.cfg,
                    metrics=metrics,
                    confusion_matrices=self.results.get("detectors", {}).get("confusion_matrices"),
                    competence_curves=self.results.get("selection", {}).get("competence"),
                    selection_history=self.results.get("selection", {}).get("drift_history"),
                )
        except Exception as e:
            logger.warning(f"MLflow logging failed: {e}")


def run_ensemble(cfg: DictConfig) -> dict[str, Any]:
    """Run CDADE pipeline and return results.

    Convenience function that initializes orchestrator and runs pipeline.

    Args:
        cfg: Hydra configuration object

    Returns:
        Dictionary containing all pipeline results
    """
    orchestrator = CDADEOrchestrator(cfg)
    return orchestrator.run()
