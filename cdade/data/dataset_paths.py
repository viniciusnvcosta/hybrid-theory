"""Helpers for resolving dataset-specific artifact paths."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any


def _dataset_name(cfg_or_name: Any) -> str:
    if isinstance(cfg_or_name, str):
        return cfg_or_name
    dataset_cfg = getattr(cfg_or_name, "dataset", None)
    if dataset_cfg is not None:
        name = getattr(dataset_cfg, "name", None)
        if isinstance(name, str) and name:
            return name
    return "sivep"


def get_dataset_artifact_paths(
    cfg_or_name: Any, project_root: Path | None = None
) -> dict[str, Path]:
    """Return dataset-specific data and result artifact paths.

    Args:
        cfg_or_name: Hydra config or dataset-name string.
        project_root: Repository root used to resolve paths.

    Returns:
        Mapping of artifact role to absolute Path.
    """
    project_root = Path(project_root or Path(__file__).resolve().parents[2])
    dataset_name = _dataset_name(cfg_or_name)

    data_dir = project_root / "data"
    injected_dir = data_dir / "injected"
    processed_dir = data_dir / "processed"

    prefix = dataset_name
    return {
        "injected_counts": injected_dir / f"{prefix}_counts_injected.parquet",
        "mask": injected_dir / f"{prefix}_counts_mask.parquet",
        "processed_counts": processed_dir / f"{prefix}_counts.parquet",
        "hierarchy": processed_dir / f"hierarchy_{prefix}.json",
    }


def _iter_datasets(
    cfg_or_name: Any, project_root: Path | None = None
) -> Iterator[tuple[str, dict[str, Path]]]:
    """Yield (dataset_name, artifact_paths) for each active dataset in cfg.

    Reads ``cfg.datasets.active`` if present; falls back to ``["sivep"]``
    when the attribute is absent (single-dataset configs stay compatible).

    Args:
        cfg_or_name: Hydra config with optional ``datasets.active`` list,
            or a bare dataset-name string for single-dataset callers.
        project_root: Repository root used to resolve paths.

    Yields:
        Tuple of (dataset_name, artifact_paths_dict).
    """
    project_root = Path(project_root or Path(__file__).resolve().parents[2])

    if isinstance(cfg_or_name, str):
        yield cfg_or_name, get_dataset_artifact_paths(cfg_or_name, project_root=project_root)
        return

    datasets_cfg = getattr(cfg_or_name, "datasets", None)
    active: list[str] = list(getattr(datasets_cfg, "active", ["sivep"]))

    for name in active:
        yield name, get_dataset_artifact_paths(name, project_root=project_root)
