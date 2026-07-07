# ABOUTME: Tests for dataset-specific artifact path resolution.
# ABOUTME: Verifies sivep and tycho artifact filenames resolve correctly.

from pathlib import Path
from types import SimpleNamespace

from cdade.data.dataset_paths import get_dataset_artifact_paths


def test_sivep_artifact_paths_use_sivep_names() -> None:
    paths = get_dataset_artifact_paths("sivep", project_root=Path("/tmp/project"))

    assert paths["injected_counts"].name == "sivep_counts_injected.parquet"
    assert paths["mask"].name == "sivep_counts_mask.parquet"
    assert paths["hierarchy"].name == "hierarchy_sivep.json"


def test_tycho_artifact_paths_use_tycho_names() -> None:
    cfg = SimpleNamespace(dataset=SimpleNamespace(name="tycho"))
    paths = get_dataset_artifact_paths(cfg, project_root=Path("/tmp/project"))

    assert paths["injected_counts"].name == "tycho_counts_injected.parquet"
    assert paths["mask"].name == "tycho_counts_mask.parquet"
    assert paths["hierarchy"].name == "hierarchy_tycho.json"
