# ABOUTME: Build AUC-PR matrices from per-dataset metrics.json files.
# ABOUTME: Loads multi-dataset metric dictionaries and constructs evaluation matrices.

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _build_auc_pr_matrix_from_dir(
    metrics_dir: Path,
) -> tuple[np.ndarray, list[str], list[str]]:
    """Build AUC-PR matrix from all per-dataset metrics.json files.

    Loads one metrics.json per dataset subdirectory and stacks AUC-PR values
    into a matrix suitable for multi-dataset hypothesis testing. Only methods
    present in every dataset's metrics.json are included, since a baseline
    can legitimately fail (and be omitted) on one dataset but not another.

    Args:
        metrics_dir: Directory containing `{dataset}/metrics.json` files.
            Each metrics.json is a dict mapping method names to metric dicts.

    Returns:
        Tuple of (auc_pr_matrix, method_names, dataset_names).
        auc_pr_matrix has shape (n_datasets, n_methods).

    Raises:
        FileNotFoundError: If no metrics.json files found under metrics_dir.
    """
    metrics_files = sorted(Path(metrics_dir).glob("*/metrics.json"))
    if not metrics_files:
        raise FileNotFoundError(f"No metrics.json files found under {metrics_dir}")

    per_dataset: list[dict] = []
    dataset_names: list[str] = []
    for mf in metrics_files:
        with open(mf) as f:
            per_dataset.append(json.load(f))
        dataset_names.append(mf.parent.name)

    method_names = sorted(set.intersection(*(set(d.keys()) for d in per_dataset)))
    auc_pr_matrix = np.array(
        [[m[method]["auc_pr"] for method in method_names] for m in per_dataset]
    )
    return auc_pr_matrix, method_names, dataset_names


def _load_scores_for_dataset(
    dataset_name: str, n_test: int = 26
) -> tuple[np.ndarray | None, np.ndarray | None, dict[str, np.ndarray] | None]:
    """Load raw scores (y_true, cdade_scores, baseline_scores) for a dataset."""
    y_true = cdade_scores = baseline_scores = None
    try:
        mask_path = Path(f"data/injected/{dataset_name}_counts_mask.parquet")
        if mask_path.exists():
            y_true = pd.read_parquet(mask_path).max(axis=1).values[-n_test:]
        blended_path = Path(f"results/selection/{dataset_name}/blended_scores.csv")
        if blended_path.exists():
            cdade_scores = pd.read_csv(blended_path, index_col=0).iloc[-n_test:].max(axis=1).values
        baseline_dir = Path(f"results/baselines/{dataset_name}")
        b_paths = sorted(baseline_dir.glob("b[1-5]_scores.npy"))
        if b_paths:
            baseline_scores = {p.stem.replace("_scores", ""): np.load(p)[-n_test:] for p in b_paths}
    except Exception as e:
        logger.warning("Could not load scores for %s: %s", dataset_name, e)
    return y_true, cdade_scores, baseline_scores
