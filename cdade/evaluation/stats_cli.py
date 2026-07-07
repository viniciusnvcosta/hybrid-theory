# ABOUTME: CLI entry-point for the stats DVC stage, dispatching single- vs multi-dataset runs.
# ABOUTME: Loads per-dataset metrics/raw scores and feeds them into run_stats_pipeline.

from __future__ import annotations

import json
import logging
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig

from cdade.evaluation.stats import run_stats_pipeline
from cdade.evaluation.stats_matrix import _build_auc_pr_matrix_from_dir, _load_scores_for_dataset

logger = logging.getLogger(__name__)


def _run_pipeline_multi_dataset(metrics_path: Path, output_dir: Path, cfg: DictConfig) -> None:
    """Run pipeline with multi-dataset matrix."""
    auc_pr_matrix, _, dataset_names = _build_auc_pr_matrix_from_dir(metrics_path)
    logger.info("Loaded AUC-PR matrix: shape %s", auc_pr_matrix.shape)
    primary = dataset_names[0]
    mask_path = Path(f"data/injected/{primary}_counts_mask.parquet")
    try:
        test_frac = getattr(getattr(cfg, "evaluation", None), "test_frac", 0.2)
        n_test = int(len(pd.read_parquet(mask_path)) * test_frac) if mask_path.exists() else 26
    except Exception as e:
        logger.warning("Failed to load mask from %s: %s, falling back to 26", mask_path, e)
        n_test = 26
    y_true, cdade_scores, baseline_scores = _load_scores_for_dataset(primary, n_test)
    with open(metrics_path / primary / "metrics.json") as f:
        primary_metrics = json.load(f)
    run_stats_pipeline(
        primary_metrics,
        alpha=0.05,
        bootstrap_n=1000,
        output_dir=output_dir,
        y_true=y_true,
        cdade_scores=cdade_scores,
        baseline_scores=baseline_scores,
        auc_pr_matrix=auc_pr_matrix,
    )


@hydra.main(
    config_path=str(Path(__file__).resolve().parents[2] / "configs"),
    config_name="config",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    """Main entry-point for DVC pipeline."""
    metrics_path = Path(cfg.evaluation.metrics_path)
    output_dir = Path(cfg.evaluation.stats_dir)
    if metrics_path.is_dir():
        _run_pipeline_multi_dataset(metrics_path, output_dir, cfg)
    else:
        with open(metrics_path) as f:
            metrics = json.load(f)
        y_true, cdade_scores, baseline_scores = _load_scores_for_dataset("sivep", 26)
        run_stats_pipeline(
            metrics,
            alpha=0.05,
            bootstrap_n=1000,
            output_dir=output_dir,
            y_true=y_true,
            cdade_scores=cdade_scores,
            baseline_scores=baseline_scores,
        )


if __name__ == "__main__":
    main()
