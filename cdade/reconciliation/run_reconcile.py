# ABOUTME: CLI entry point for the reconciliation pipeline stage (DVC: reconcile).
# ABOUTME: Writes reconciled detector scores to results/reconciliation/{dataset}/.

"""CLI entry point for reconciliation pipeline stage.

Reads detector outputs from results/detectors/{dataset}/,
applies hierarchical reconciliation using the selected method,
and writes reconciled scores to results/reconciliation/{dataset}/.
"""

import json
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig

from cdade.data.dataset_paths import get_dataset_artifact_paths

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def run_reconcile(cfg: DictConfig, dataset_name: str | None = None) -> dict:
    """Reconcile detector outputs for one dataset.

    Args:
        cfg: Hydra configuration object.
        dataset_name: Dataset name override. If None, inferred from cfg.

    Returns:
        Dictionary containing reconciliation results.
    """
    import cdade.reconciliation.bottom_up  # noqa: F401
    import cdade.reconciliation.evt  # noqa: F401
    import cdade.reconciliation.identity  # noqa: F401
    import cdade.reconciliation.min_t  # noqa: F401
    from cdade.reconciliation import get_reconciler

    if dataset_name is None:
        from cdade.data.dataset_paths import _dataset_name as _dn

        dataset_name = _dn(cfg)

    artifact_paths = get_dataset_artifact_paths(dataset_name, project_root=_PROJECT_ROOT)
    hierarchy_path = artifact_paths["hierarchy"]
    if not hierarchy_path.exists():
        raise FileNotFoundError(f"Hierarchy spec not found: {hierarchy_path}")

    with open(hierarchy_path) as f:
        spec = json.load(f)

    detector_dir = _PROJECT_ROOT / "results" / "detectors" / dataset_name
    leaf_forecasts = pd.read_csv(detector_dir / "leaf_forecasts.csv")

    reconciler_name = cfg.reconciliation.name
    reconciler_cls = get_reconciler(reconciler_name)
    reconciler = reconciler_cls(cfg.reconciliation)

    reconciler.fit(spec, leaf_forecasts)
    reconciled_leaves, reconciled_aggregate, residuals = reconciler.reconcile(leaf_forecasts)

    output_dir = _PROJECT_ROOT / "results" / "reconciliation" / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    reconciled_leaves.to_csv(output_dir / "leaf_forecasts_reconciled.csv")
    reconciled_aggregate.to_csv(output_dir / "aggregate_forecasts.csv")
    residuals.to_csv(output_dir / "residuals.csv")

    spec["S"] = spec["leaves"]
    with open(output_dir / "hierarchy.json", "w") as f:
        json.dump(spec, f, indent=2)

    return {
        "coherent_scores": reconciled_leaves,
        "reconciled_count": len(reconciled_leaves.columns),
    }


@hydra.main(
    config_path=str(_PROJECT_ROOT / "configs"),
    config_name="config",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    """CLI entry point for the reconcile stage — loops over datasets.active."""
    from cdade.data.dataset_paths import _iter_datasets

    for dataset_name, _paths in _iter_datasets(cfg, project_root=_PROJECT_ROOT):
        run_reconcile(cfg, dataset_name=dataset_name)


if __name__ == "__main__":
    main()
