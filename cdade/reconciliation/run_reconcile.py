"""CLI entry point for reconciliation pipeline stage."""

import json
from pathlib import Path

import pandas as pd
from omegaconf import DictConfig


def run_reconcile(cfg: DictConfig) -> dict:
    """Reconcile detector outputs using selected method.

    Args:
        cfg: Hydra configuration object

    Returns:
        Dictionary containing reconciliation results
    """
    from cdade.reconciliation import get_reconciler

    # Load hierarchy spec from processed data
    hierarchy_path = Path("../data/processed/hierarchy_sivep.json")
    if hierarchy_path.exists():
        with open(hierarchy_path) as f:
            spec = json.load(f)
    else:
        raise FileNotFoundError(f"Hierarchy spec not found: {hierarchy_path}")

    # Load detector outputs
    detector_dir = Path("../results/detectors")
    leaf_forecasts = pd.read_csv(detector_dir / "leaf_forecasts.csv", index_col=0)

    # Select reconciler
    reconciler_name = cfg.reconciliation.name
    reconciler = get_reconciler(reconciler_name)

    # Fit and reconcile
    reconciler.fit(spec, leaf_forecasts)
    reconciled_leaves, reconciled_aggregate, residuals = reconciler.reconcile(leaf_forecasts)

    # Save results
    output_dir = Path("../results/reconciliation")
    output_dir.mkdir(parents=True, exist_ok=True)

    reconciled_leaves.to_csv(output_dir / "leaf_forecasts_reconciled.csv")
    reconciled_aggregate.to_csv(output_dir / "aggregate_forecasts.csv")
    residuals.to_csv(output_dir / "residuals.csv")

    # Save hierarchy with S matrix
    spec["S"] = spec["leaves"]  # Simplified: store S as dict
    with open(output_dir / "hierarchy.json", "w") as f:
        json.dump(spec, f, indent=2)

    return {
        "coherent_scores": reconciled_leaves,
        "reconciled_count": len(reconciled_leaves.columns),
    }
