# ABOUTME: Build AUC-PR matrices from per-dataset metrics.json files.
# ABOUTME: Loads multi-dataset metric dictionaries and constructs evaluation matrices.

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _build_auc_pr_matrix_from_dir(
    metrics_dir: Path,
) -> tuple[np.ndarray, list[str], list[str]]:
    """Build AUC-PR matrix from all per-dataset metrics.json files.

    Loads one metrics.json per dataset subdirectory and stacks AUC-PR values
    into a matrix suitable for multi-dataset hypothesis testing.

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

    method_names = sorted(per_dataset[0].keys())
    auc_pr_matrix = np.array(
        [[m[method]["auc_pr"] for method in method_names] for m in per_dataset]
    )
    return auc_pr_matrix, method_names, dataset_names
