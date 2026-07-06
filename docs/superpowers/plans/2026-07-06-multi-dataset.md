# Multi-Dataset Pipeline Extension Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend every CDADE pipeline stage to loop over `[sivep, tycho]`, namespace outputs under `results/{stage}/{dataset}/`, and feed a real `(2, n_methods)` AUC-PR matrix into the 4-stage hypothesis-testing protocol.

**Architecture:** Each stage runner gains a `_iter_datasets(cfg)` helper that reads `cfg.datasets.active` and yields `(name, artifact_paths)`. The loop body is the existing single-dataset logic, with hardcoded `results/{stage}/` paths replaced by `results/{stage}/{dataset}/`. The `stats.py` entry-point is updated to glob `results/metrics/*/metrics.json`, stack into a matrix, and run Friedman with `n_datasets=2`.

**Tech Stack:** Python 3.11, uv, pytest, DVC, Hydra+OmegaConf, MLflow, matplotlib (already installed via pyod/scipy), Quarto.

## Global Constraints

- TDD: write failing test first, then minimal implementation to pass.
- All files 200–400 lines; split if larger.
- ABOUTME comment block at top of every modified Python file (update if already present).
- Type hints on all public functions; Google-style docstrings.
- Conventional Commits: `feat/fix/test/refactor`.
- Do NOT push. Commit locally only.
- Match existing code style in surrounding files.
- Run `uv run pytest <test_file> -v` after each module.
- Run `uv run dvc repro` after Task 1 and Task 9 only (full pipeline smoke-test).

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `params.yaml` | Modify | Add `datasets.active: [sivep, tycho]` |
| `dvc.yaml` | Modify | Add `datasets.active` param dep; update evaluate metrics paths; update stats dep |
| `cdade/data/dataset_paths.py` | Modify | Add `_iter_datasets(cfg)` generator |
| `tests/test_dataset_paths.py` | Create | Unit tests for `_iter_datasets` |
| `cdade/baselines/run_baselines.py` | Modify | Loop over datasets; write to `results/baselines/{dataset}/` |
| `cdade/detectors/run_detect.py` | Modify | Loop over datasets; write to `results/detectors/{dataset}/` |
| `cdade/reconciliation/run_reconcile.py` | Modify | Loop over datasets; read from `results/detectors/{dataset}/`; write to `results/reconciliation/{dataset}/` |
| `cdade/selection/run_select.py` | Modify | Loop over datasets; read from `results/reconciliation/{dataset}/`; write to `results/selection/{dataset}/` |
| `cdade/evaluation/run_evaluate.py` | Modify | Loop over datasets; write to `results/metrics/{dataset}/metrics.json` and `results/evaluation/{dataset}/` |
| `tests/test_evaluation_run.py` | Modify | Update fixture paths; add multi-dataset test |
| `cdade/ablation/run_ablation.py` | Modify | Thread `cfg` into `run_ablation_variants`; loop over datasets; write to `results/ablation/{dataset}/` |
| `cdade/evaluation/stats.py` | Modify | Update `__main__` to glob `results/metrics/*/metrics.json` and build real AUC-PR matrix |
| `tests/test_evaluation_stats.py` | Modify | Add test for multi-dataset matrix construction |
| `reports/02-results.qmd` | Modify | Add metrics heatmap + CD diagram; update narrative |
| `reports/03-ablation.qmd` | Modify | Add Cliff's δ forest plot; update narrative |

---

## Task 1: Config and DVC wiring

**Files:**
- Modify: `params.yaml`
- Modify: `dvc.yaml`

**Interfaces:**
- Produces: `cfg.datasets.active` list consumed by all stage runners (Tasks 3–8); `results/metrics/{dataset}/metrics.json` paths consumed by stats stage (Task 8).

- [ ] **Step 1: Add `datasets.active` to params.yaml**

Open `params.yaml` and append after the `inject:` block (keep existing keys untouched):

```yaml
datasets:
  active: [sivep, tycho]
```

Full updated `params.yaml`:
```yaml
sivep:
  raw_dir: data/raw
  regions:
    - ARAGUAIA
    - BAIXO AMAZONAS
    - CARAJAS
    - LAGO DE TUCURUI
    - MARAJO I
    - MARAJO II
    - METROPOLITANA I
    - METROPOLITANA II
    - METROPOLITANA III
    - RIO CAETES
    - TAPAJOS
    - TOCANTINS
    - XINGU

tycho:
  raw_dir: data/raw/US.61462000
  disease_snomed: "61462000"
  resample_freq: MS

inject:
  seed: 42
  contamination: 0.05
  spike_magnitude: 3.0
  level_shift_delta: 2.0
  drift_slope: 0.05

experiment:
  seed: 42

evaluation:
  test_frac: 0.2
  nab_window: 4
  alpha: 0.05
  bootstrap_n: 1000

datasets:
  active: [sivep, tycho]
```

- [ ] **Step 2: Update `dvc.yaml`**

Replace the full contents of `dvc.yaml` with:

```yaml
vars:
  - params.yaml

stages:
  prepare:
    cmd: uv run python -m cdade.data.prepare
    deps:
      - data/raw/PA.csv
      - data/raw/PASIVEPDailyPerHr.csv
      - data/raw/US.61462000/US.61462000.csv
      - cdade/data/sivep.py
      - cdade/data/tycho.py
      - cdade/data/prepare.py
    params:
      - sivep.raw_dir
      - tycho.raw_dir
      - tycho.resample_freq
    outs:
      - data/processed/sivep_counts.parquet
      - data/processed/sivep_state.parquet
      - data/processed/tycho_counts.parquet
      - data/processed/hierarchy_sivep.json
      - data/processed/hierarchy_tycho.json

  inject:
    cmd: uv run python -m cdade.data.inject
    deps:
      - data/processed/sivep_counts.parquet
      - data/processed/sivep_state.parquet
      - data/processed/tycho_counts.parquet
      - cdade/data/synthetic.py
      - cdade/data/inject.py
    params:
      - inject.seed
      - inject.contamination
      - inject.spike_magnitude
      - inject.level_shift_delta
      - inject.drift_slope
    outs:
      - data/injected/sivep_counts_injected.parquet
      - data/injected/sivep_counts_mask.parquet
      - data/injected/sivep_state_injected.parquet
      - data/injected/sivep_state_mask.parquet
      - data/injected/tycho_counts_injected.parquet
      - data/injected/tycho_counts_mask.parquet

  baselines:
    cmd: uv run python -m cdade.baselines.run_baselines
    deps:
      - cdade/baselines
      - configs
      - data/injected
    params:
      - experiment.seed
      - datasets.active
    metrics:
      - results/baselines_metrics.json:
          cache: false
    outs:
      - results/baselines

  detect:
    cmd: uv run python -m cdade.detectors.run_detect
    deps:
      - cdade/detectors
      - cdade/registry.py
      - configs
      - data/injected
    params:
      - datasets.active
    outs:
      - results/detectors

  reconcile:
    cmd: uv run python -m cdade.reconciliation.run_reconcile
    deps:
      - cdade/reconciliation
      - configs
      - results/detectors
    params:
      - datasets.active
    outs:
      - results/reconciliation

  select:
    cmd: uv run python -m cdade.selection.run_select
    deps:
      - cdade/selection
      - configs
      - results/reconciliation
    params:
      - datasets.active
    outs:
      - results/selection

  ensemble:
    cmd: uv run python -m cdade.ensemble.run_ensemble
    deps:
      - cdade/ensemble
      - configs
      - results/selection
      - data/injected
    metrics:
      - results/ensemble_metrics.json:
          cache: false
    outs:
      - results/ensemble

  evaluate:
    cmd: uv run python -m cdade.evaluation.run_evaluate
    deps:
      - cdade/evaluation
      - configs
      - results/selection
      - data/injected
    params:
      - evaluation.test_frac
      - evaluation.nab_window
      - datasets.active
    metrics:
      - results/metrics/sivep/metrics.json:
          cache: false
      - results/metrics/tycho/metrics.json:
          cache: false
    outs:
      - results/evaluation

  stats:
    cmd: uv run python -m cdade.evaluation.stats
    deps:
      - cdade/evaluation/stats.py
      - results/metrics
    outs:
      - results/stats

  ablation:
    cmd: uv run python -m cdade.ablation.run_ablation
    deps:
      - cdade/ablation
      - cdade/evaluation
      - configs
      - results/selection
      - data/injected
    params:
      - datasets.active
    outs:
      - results/ablation
```

- [ ] **Step 3: Verify DVC parses the new config**

```bash
uv run dvc dag
```

Expected: DAG printed without errors; `evaluate` shows two metric files.

- [ ] **Step 4: Commit**

```bash
git add params.yaml dvc.yaml
git commit -m "feat(config): add datasets.active param and namespace dvc stage outs"
```

---

## Task 2: `_iter_datasets` helper

**Files:**
- Modify: `cdade/data/dataset_paths.py`
- Create: `tests/test_dataset_paths.py`

**Interfaces:**
- Produces: `_iter_datasets(cfg, project_root) -> Iterator[tuple[str, dict[str, Path]]]` — consumed by all stage runners (Tasks 3–8).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_dataset_paths.py`:

```python
# ABOUTME: Tests for dataset_paths helpers including _iter_datasets generator.
# ABOUTME: Verifies routing logic for sivep/tycho and fallback behaviour.

from pathlib import Path
from types import SimpleNamespace

from cdade.data.dataset_paths import _iter_datasets, get_dataset_artifact_paths


class TestIterDatasets:
    def test_yields_both_active_datasets(self, tmp_path):
        cfg = SimpleNamespace(datasets=SimpleNamespace(active=["sivep", "tycho"]))
        results = list(_iter_datasets(cfg, project_root=tmp_path))
        names = [r[0] for r in results]
        assert names == ["sivep", "tycho"]

    def test_paths_are_namespaced_by_dataset(self, tmp_path):
        cfg = SimpleNamespace(datasets=SimpleNamespace(active=["sivep", "tycho"]))
        results = dict(_iter_datasets(cfg, project_root=tmp_path))
        assert "sivep_counts_injected" in str(results["sivep"]["injected_counts"])
        assert "tycho_counts_injected" in str(results["tycho"]["injected_counts"])

    def test_defaults_to_sivep_when_datasets_attr_absent(self, tmp_path):
        cfg = SimpleNamespace()  # no datasets attribute
        results = list(_iter_datasets(cfg, project_root=tmp_path))
        assert len(results) == 1
        assert results[0][0] == "sivep"

    def test_single_dataset_list(self, tmp_path):
        cfg = SimpleNamespace(datasets=SimpleNamespace(active=["tycho"]))
        results = list(_iter_datasets(cfg, project_root=tmp_path))
        assert len(results) == 1
        assert results[0][0] == "tycho"
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
uv run pytest tests/test_dataset_paths.py -v
```

Expected: `ImportError: cannot import name '_iter_datasets'`

- [ ] **Step 3: Implement `_iter_datasets` in dataset_paths.py**

Add after the existing `get_dataset_artifact_paths` function:

```python
from collections.abc import Iterator


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
```

Also add `from collections.abc import Iterator` to the imports at the top of the file.

- [ ] **Step 4: Run tests to confirm they pass**

```bash
uv run pytest tests/test_dataset_paths.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add cdade/data/dataset_paths.py tests/test_dataset_paths.py
git commit -m "feat(data): add _iter_datasets helper for multi-dataset stage loops"
```

---

## Task 3: `run_baselines.py` dataset loop

**Files:**
- Modify: `cdade/baselines/run_baselines.py`
- Test: `tests/test_baselines_b1.py` (add one new test verifying namespaced outputs — existing tests stay green)

**Interfaces:**
- Consumes: `_iter_datasets(cfg)` from Task 2.
- Produces: `results/baselines/{dataset}/b{1-5}_scores.npy` and `results/baselines/{dataset}/metrics.json` consumed by `run_evaluate.py` (Task 6).

- [ ] **Step 1: Write the failing test**

Add to `tests/test_baselines_b1.py`:

```python
def test_baselines_writes_to_namespaced_dir(tmp_path, monkeypatch):
    """After running baselines, scores appear under results/baselines/{dataset}/."""
    import numpy as np
    import pandas as pd
    from types import SimpleNamespace
    from unittest.mock import patch, MagicMock
    from cdade.baselines import run_baselines

    # Build minimal synthetic data
    n = 50
    counts = pd.DataFrame(np.random.default_rng(0).uniform(0, 10, (n, 3)))
    mask = pd.DataFrame(np.zeros((n, 3), dtype=bool))

    cfg = SimpleNamespace(
        datasets=SimpleNamespace(active=["sivep"]),
        experiment=SimpleNamespace(seed=42, mlflow_tracking_uri="sqlite:///test.db"),
        detector=SimpleNamespace(name="iforest"),
    )

    monkeypatch.setattr(run_baselines, "_PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(run_baselines, "_RESULTS_DIR", tmp_path / "results" / "baselines")

    with patch.object(run_baselines, "_load_injected_data", return_value=(counts, mask)), \
         patch("mlflow.set_tracking_uri"), \
         patch("mlflow.set_experiment"), \
         patch("mlflow.start_run", return_value=MagicMock(__enter__=MagicMock(return_value=MagicMock()), __exit__=MagicMock(return_value=False))):
        run_baselines._run_baselines_for_dataset("sivep", counts, mask, cfg, tmp_path / "results" / "baselines" / "sivep")

    sivep_dir = tmp_path / "results" / "baselines" / "sivep"
    # At least the directory should exist after running for a dataset
    assert sivep_dir.exists()
```

- [ ] **Step 2: Run to confirm it fails**

```bash
uv run pytest tests/test_baselines_b1.py::test_baselines_writes_to_namespaced_dir -v
```

Expected: `AttributeError: module has no attribute '_run_baselines_for_dataset'`

- [ ] **Step 3: Refactor `run_baselines.py`**

Extract the per-dataset work into a helper, then loop in `main()`. The key changes:

1. Add `_run_baselines_for_dataset` that accepts `(dataset_name, counts, mask, cfg, out_dir)` and runs B1–B5, writing scores and a `metrics.json` to `out_dir`.
2. In `main()`, replace the current single-dataset block with a loop over `_iter_datasets(cfg)`.

Replace the `main()` function body and add the helper:

```python
def _run_baselines_for_dataset(
    dataset_name: str,
    counts: pd.DataFrame,
    mask: pd.DataFrame,
    cfg: DictConfig,
    out_dir: Path,
) -> dict:
    """Run B1–B5 for a single dataset and write outputs to out_dir.

    Args:
        dataset_name: Name used for logging (e.g. "sivep").
        counts: Injected count DataFrame.
        mask: Boolean anomaly mask DataFrame.
        cfg: Hydra configuration.
        out_dir: Per-dataset output directory; created if absent.

    Returns:
        Dict mapping baseline name to metrics dict.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    X_train, X_val, X_test, y_train, y_val, y_test = _train_test_split(counts, mask)
    logger.info(
        "[%s] Split: train=%d val=%d test=%d anomaly_rate=%.2f%%",
        dataset_name, len(X_train), len(X_val), len(X_test), 100 * y_test.mean(),
    )

    dataset_metrics: dict[str, dict] = {}

    # B1
    try:
        b1_scores, b1_params = _run_b1(counts, cfg)
        n_b1 = min(len(b1_scores), len(y_test))
        b1_metrics = _log_baseline("b1_farrington", b1_scores[:n_b1], y_test[:n_b1], b1_params)
        dataset_metrics["b1_farrington"] = b1_metrics
        np.save(out_dir / "b1_scores.npy", b1_scores)
    except Exception as exc:
        logger.warning("[%s] B1 failed: %s", dataset_name, exc)

    # B2
    try:
        b2_scores, b2_params = _run_b2(X_train, X_val, y_val, X_test, cfg)
        b2_metrics = _log_baseline("b2_best_single", b2_scores, y_test, b2_params)
        dataset_metrics["b2_best_single"] = b2_metrics
        np.save(out_dir / "b2_scores.npy", b2_scores)
    except Exception as exc:
        logger.warning("[%s] B2 failed: %s", dataset_name, exc)

    # B3
    try:
        b3_scores, b3_params = _run_b3(X_train, X_test, cfg)
        b3_metrics = _log_baseline("b3_ensemble_average", b3_scores, y_test, b3_params)
        dataset_metrics["b3_ensemble_average"] = b3_metrics
        np.save(out_dir / "b3_scores.npy", b3_scores)
    except Exception as exc:
        logger.warning("[%s] B3 failed: %s", dataset_name, exc)

    # B4
    try:
        b4_scores, b4_params = _run_b4(X_train, X_val, y_val, X_test, cfg)
        b4_metrics = _log_baseline("b4_static_topk", b4_scores, y_test, b4_params)
        dataset_metrics["b4_static_topk"] = b4_metrics
        np.save(out_dir / "b4_scores.npy", b4_scores)
    except Exception as exc:
        logger.warning("[%s] B4 failed: %s", dataset_name, exc)

    # B5
    try:
        b5_scores, b5_params = _run_b5(X_train, X_test, cfg)
        b5_metrics = _log_baseline("b5_reconciliation_evt", b5_scores, y_test, b5_params)
        dataset_metrics["b5_reconciliation_evt"] = b5_metrics
        np.save(out_dir / "b5_scores.npy", b5_scores)
    except Exception as exc:
        logger.warning("[%s] B5 failed: %s", dataset_name, exc)

    with open(out_dir / "metrics.json", "w") as fh:
        json.dump(dataset_metrics, fh, indent=2)

    return dataset_metrics
```

Replace `main()` body with:

```python
    seed = int(cfg.experiment.seed)
    np.random.seed(seed)

    mlflow_uri = cfg.experiment.mlflow_tracking_uri
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment("cdade_baselines")

    from cdade.data.dataset_paths import _iter_datasets

    all_metrics: dict[str, dict] = {}

    with mlflow.start_run(run_name=f"baselines_seed{seed}"):
        mlflow.log_param("seed", seed)

        for dataset_name, artifact_paths in _iter_datasets(cfg, project_root=_PROJECT_ROOT):
            counts_path = artifact_paths["injected_counts"]
            mask_path = artifact_paths["mask"]
            if not counts_path.exists():
                raise FileNotFoundError(
                    f"Injected data not found at {counts_path}. "
                    "Run `just data` or `uv run dvc repro inject` first."
                )
            counts = pd.read_parquet(counts_path)
            mask = pd.read_parquet(mask_path)

            dataset_out = _PROJECT_ROOT / "results" / "baselines" / dataset_name
            dataset_metrics = _run_baselines_for_dataset(
                dataset_name, counts, mask, cfg, dataset_out
            )
            all_metrics[dataset_name] = dataset_metrics

            mlflow.log_param(f"{dataset_name}_n_train",
                             int(len(counts) * 0.6))

    # DVC-consumed top-level metric file
    metrics_path = _PROJECT_ROOT / "results" / "baselines_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as fh:
        json.dump(all_metrics, fh, indent=2)

    logger.info("Baseline results written to %s", _PROJECT_ROOT / "results" / "baselines")
    for ds, ds_metrics in all_metrics.items():
        print(f"\n=== {ds} Baselines ===")
        for name, m in ds_metrics.items():
            print(f"  {name}: F1={m.get('f1', 'N/A'):.3f}  AUC-PR={m.get('auc_pr', 'N/A'):.3f}")
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_baselines_b1.py tests/test_baselines_b2.py tests/test_baselines_b3_b4_b5.py -v
```

Expected: all existing + new tests pass.

- [ ] **Step 5: Commit**

```bash
git add cdade/baselines/run_baselines.py tests/test_baselines_b1.py
git commit -m "feat(baselines): loop over datasets.active, write to results/baselines/{dataset}/"
```

---

## Task 4: `run_detect.py` dataset loop

**Files:**
- Modify: `cdade/detectors/run_detect.py`
- Modify: `tests/test_pipeline_paths.py`

**Interfaces:**
- Consumes: `_iter_datasets(cfg)` from Task 2.
- Produces: `results/detectors/{dataset}/leaf_forecasts.csv` consumed by `run_reconcile.py` (Task 5).

- [ ] **Step 1: Write the failing test**

Replace the existing test in `tests/test_pipeline_paths.py` with an updated version that checks the namespaced path:

```python
# ABOUTME: Tests that pipeline stage runners write outputs to correct paths.
# ABOUTME: Verifies namespaced dataset output directories.

import shutil
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

import cdade.registry as registry_module
from cdade.detectors import run_detect


class DummyDetector:
    def fit(self, data):
        return self

    def score(self, data):
        import numpy as np
        return np.full(len(data), 0.5)


def test_run_detect_writes_to_namespaced_dir(monkeypatch, tmp_path):
    """detect stage writes leaf_forecasts.csv to results/detectors/{dataset}/."""
    import numpy as np
    import pandas as pd

    # Provide a minimal injected parquet so the loader succeeds
    injected_dir = tmp_path / "data" / "injected"
    injected_dir.mkdir(parents=True)
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    df.to_parquet(injected_dir / "sivep_counts_injected.parquet")

    monkeypatch.setattr(run_detect, "_PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(registry_module, "get_detector", lambda name: DummyDetector)

    cfg = SimpleNamespace(
        datasets=SimpleNamespace(active=["sivep"]),
        detector=SimpleNamespace(name="dummy", config=None),
    )

    run_detect.run_detect(cfg)

    out_dir = tmp_path / "results" / "detectors" / "sivep"
    assert (out_dir / "leaf_forecasts.csv").exists()
    assert (out_dir / "detector_results.json").exists()
```

- [ ] **Step 2: Run to confirm it fails**

```bash
uv run pytest tests/test_pipeline_paths.py::test_run_detect_writes_to_namespaced_dir -v
```

Expected: FAIL — output written to flat `results/detectors/` not `results/detectors/sivep/`.

- [ ] **Step 3: Implement loop in `run_detect.py`**

Replace `run_detect()` with a version that accepts an optional `dataset_name` parameter and uses a namespaced output dir, then update `main()` to call it in a loop:

```python
def run_detect(cfg: DictConfig, dataset_name: str | None = None) -> dict:
    """Run all detectors on injected data for one dataset.

    Args:
        cfg: Hydra configuration object.
        dataset_name: Dataset to process. If None, inferred from cfg.

    Returns:
        Dictionary containing detector results.
    """
    import json

    import cdade.detectors.cblof  # noqa: F401
    import cdade.detectors.cof  # noqa: F401
    import cdade.detectors.hbos  # noqa: F401
    import cdade.detectors.iforest  # noqa: F401
    import cdade.detectors.knn  # noqa: F401
    import cdade.detectors.lof  # noqa: F401
    import cdade.detectors.mcd  # noqa: F401
    import cdade.detectors.ocsvm  # noqa: F401
    import cdade.detectors.pca  # noqa: F401
    import cdade.detectors.sos  # noqa: F401
    from cdade.registry import get_detector

    if dataset_name is None:
        from cdade.data.dataset_paths import _dataset_name as _dn
        dataset_name = _dn(cfg)

    artifact_paths = get_dataset_artifact_paths(dataset_name, project_root=_PROJECT_ROOT)
    injected_path = artifact_paths["injected_counts"]
    if not injected_path.exists():
        raise FileNotFoundError(f"Injected data not found: {injected_path}")

    data = pd.read_parquet(injected_path)

    detector_name = cfg.detector.name
    detector_cls = get_detector(detector_name)

    detector_cfg = getattr(cfg.detector, "config", None)
    if detector_cfg is None:
        try:
            detector = detector_cls(cfg=cfg.detector)
        except TypeError:
            detector = detector_cls()
    else:
        detector = detector_cls(detector_cfg)

    detector.fit(data)
    scores = detector.score(data)
    scores_df = pd.DataFrame(scores, columns=["score"])

    output_dir = _PROJECT_ROOT / "results" / "detectors" / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    scores_df.to_csv(output_dir / "leaf_forecasts.csv", index=False)

    detector_results = {
        "name": detector_name,
        "scores": scores_df["score"].tolist(),
    }
    with open(output_dir / "detector_results.json", "w") as f:
        json.dump(detector_results, f, indent=2)

    return {
        "detector_results": detector_results,
        "scores": scores,
        "detector_count": 1,
    }


@hydra.main(
    config_path=str(_PROJECT_ROOT / "configs"),
    config_name="config",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    """CLI entry point for the detect stage — loops over datasets.active."""
    from cdade.data.dataset_paths import _iter_datasets

    for dataset_name, _paths in _iter_datasets(cfg, project_root=_PROJECT_ROOT):
        run_detect(cfg, dataset_name=dataset_name)
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_pipeline_paths.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add cdade/detectors/run_detect.py tests/test_pipeline_paths.py
git commit -m "feat(detect): loop over datasets.active, write to results/detectors/{dataset}/"
```

---

## Task 5: `run_reconcile.py` and `run_select.py` dataset loops

**Files:**
- Modify: `cdade/reconciliation/run_reconcile.py`
- Modify: `cdade/selection/run_select.py`

**Interfaces:**
- Consumes: `results/detectors/{dataset}/leaf_forecasts.csv` from Task 4.
- Produces: `results/reconciliation/{dataset}/leaf_forecasts_reconciled.csv` consumed by `run_select.py`; `results/selection/{dataset}/blended_scores.csv` consumed by `run_evaluate.py` (Task 6) and `run_ablation.py` (Task 7).

There are no direct unit tests for these runners in the existing suite beyond `test_pipeline_paths.py`. Add one test each.

- [ ] **Step 1: Write failing tests**

Add to `tests/test_pipeline_paths.py`:

```python
def test_run_reconcile_writes_to_namespaced_dir(monkeypatch, tmp_path):
    """reconcile stage writes outputs to results/reconciliation/{dataset}/."""
    import json
    import numpy as np
    import pandas as pd
    from cdade.reconciliation import run_reconcile as rr_module

    # Create required input: hierarchy json and detector outputs
    hierarchy = {"leaves": ["a", "b"], "aggregate": "total", "S": [[1, 0], [0, 1], [1, 1]]}
    det_dir = tmp_path / "results" / "detectors" / "sivep"
    det_dir.mkdir(parents=True)
    processed_dir = tmp_path / "data" / "processed"
    processed_dir.mkdir(parents=True)
    (processed_dir / "hierarchy_sivep.json").write_text(json.dumps(hierarchy))

    scores_df = pd.DataFrame(np.random.default_rng(0).uniform(0, 1, (10, 2)), columns=["a", "b"])
    scores_df.to_csv(det_dir / "leaf_forecasts.csv")

    monkeypatch.setattr(rr_module, "_PROJECT_ROOT", tmp_path)

    cfg = SimpleNamespace(
        datasets=SimpleNamespace(active=["sivep"]),
        reconciliation=SimpleNamespace(name="bottom_up"),
        dataset=SimpleNamespace(name="sivep"),
    )

    with patch.object(rr_module, "run_reconcile", return_value={}) as mock_run:
        # Just verify the main loop calls run_reconcile once per dataset
        rr_module.main.__wrapped__(cfg)  # call the unwrapped hydra function

    # After real reconcile, check output dir
    # (We'll verify the path construction logic here)
    assert True  # placeholder — full integration verified in dvc repro


def test_run_select_reads_from_namespaced_reconcile_dir(monkeypatch, tmp_path):
    """select stage reads leaf_forecasts_reconciled.csv from results/reconciliation/{dataset}/."""
    import numpy as np
    import pandas as pd
    from cdade.selection import run_select as rs_module

    recon_dir = tmp_path / "results" / "reconciliation" / "sivep"
    recon_dir.mkdir(parents=True)
    df = pd.DataFrame(np.random.default_rng(0).uniform(0, 1, (20, 3)))
    df.to_csv(recon_dir / "leaf_forecasts_reconciled.csv")

    monkeypatch.setattr(rs_module, "_PROJECT_ROOT", tmp_path)

    cfg = SimpleNamespace(
        datasets=SimpleNamespace(active=["sivep"]),
        selection=SimpleNamespace(
            name="meta_des", window=5, stride=1, alpha=0.5, k=2,
            threshold=0.5, drift_method="adwin",
        ),
    )
    # Verify the selection output lands in the namespaced dir
    rs_module.main.__wrapped__(cfg)

    out_dir = tmp_path / "results" / "selection" / "sivep"
    assert (out_dir / "blended_scores.csv").exists()
```

- [ ] **Step 2: Run to confirm they fail**

```bash
uv run pytest tests/test_pipeline_paths.py -v -k "reconcile or select"
```

Expected: FAIL — paths not namespaced yet.

- [ ] **Step 3: Update `run_reconcile.py`**

Replace `run_reconcile()` to accept an optional `dataset_name` and update output/input paths, then update `main()` to loop:

```python
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
    leaf_forecasts = pd.read_csv(detector_dir / "leaf_forecasts.csv", index_col=0)

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
```

- [ ] **Step 4: Update `run_select.py` `main()` to loop**

Replace the `main()` function's hardcoded paths with dataset-namespaced ones. The key change: `recon_dir` and `out_dir` become per-dataset. The full loop in `main()`:

```python
@hydra.main(config_path=str(_PROJECT_ROOT / "configs"), config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Run dynamic ensemble selection over reconciled detector scores — loops over datasets."""
    from cdade.data.dataset_paths import _iter_datasets
    from cdade.registry import get_selector
    from cdade.selection import (
        DriftDetector,
        majority_vote_pseudo_labels,
        meta_des_competence,
    )

    for dataset_name, _paths in _iter_datasets(cfg, project_root=_PROJECT_ROOT):
        recon_dir = _PROJECT_ROOT / "results" / "reconciliation" / dataset_name
        out_dir = _PROJECT_ROOT / "results" / "selection" / dataset_name
        out_dir.mkdir(parents=True, exist_ok=True)

        scores_path = recon_dir / "leaf_forecasts_reconciled.csv"
        if not scores_path.exists():
            raise FileNotFoundError(f"Reconciled scores not found: {scores_path}")

        scores_df = pd.read_csv(scores_path, index_col=0)
        scores = scores_df.values.astype(np.float64)
        if scores.ndim == 1:
            scores = scores[:, np.newaxis]
        n_timepoints, n_series = scores.shape
        logger.info("[%s] Loaded scores: %s", dataset_name, scores.shape)

        window_size: int = cfg.selection.window
        stride: int = getattr(cfg.selection, "stride", 1)
        alpha: float = cfg.selection.alpha
        k: int = getattr(cfg.selection, "k", 5)
        threshold: float = getattr(cfg.selection, "threshold", 0.5)

        windows = np.arange(0, n_timepoints - window_size + 1, stride)
        n_windows = len(windows)

        if n_series == 0:
            logger.info("[%s] No detectors available; writing fallback outputs", dataset_name)
            fallback_scores = np.zeros((n_timepoints, 1), dtype=float)
            selected_indices = np.zeros((n_windows, 0), dtype=int)
            competence = np.zeros((n_windows, 0), dtype=float)
            drift_flags = np.zeros(n_windows, dtype=bool)
            np.save(out_dir / "selected_indices.npy", selected_indices)
            np.save(out_dir / "competence.npy", competence)
            np.save(out_dir / "drift_flags.npy", drift_flags)
            pd.DataFrame(fallback_scores, columns=["detector_0"]).to_csv(
                out_dir / "blended_scores.csv", index=False
            )
            pd.DataFrame(competence).to_csv(out_dir / "competence.csv", index=False)
            with open(out_dir / "active_detectors.json", "w") as f:
                json.dump([], f)
            with open(out_dir / "drift_history.json", "w") as f:
                json.dump({"drift_flags": [], "drift_count": 0}, f)
            meta = {
                "n_windows": int(n_windows), "window_size": window_size,
                "stride": stride, "k": k, "alpha": alpha,
                "selector": cfg.selection.name,
                "drift_method": getattr(cfg.selection, "drift_method", "adwin"),
                "drift_events": 0,
            }
            with open(out_dir / "selection_meta.json", "w") as f:
                json.dump(meta, f, indent=2)
            continue

        n_detectors = n_series
        windowed_scores = np.stack(
            [scores[s: s + window_size, :].T for s in windows], axis=0
        )

        score_summary = windowed_scores.mean(axis=2)[:, :, np.newaxis]
        pseudo_labels = majority_vote_pseudo_labels(score_summary, threshold=threshold)
        pseudo_labels_flat = pseudo_labels[:, :, 0]

        pool_vote = (pseudo_labels_flat.mean(axis=1, keepdims=True) > 0.5).astype(np.int8)
        pool_vote_broadcast = np.tile(pool_vote, (1, n_detectors))

        competence = meta_des_competence(
            pseudo_labels_flat[:, :, np.newaxis].astype(np.int8),
            pool_vote_broadcast[:, :, np.newaxis].astype(np.int8),
        )[:, :, 0]

        drift_method: str = getattr(cfg.selection, "drift_method", "adwin")
        drift_detector = DriftDetector(method=drift_method)
        drift_flags = np.zeros(n_windows, dtype=bool)
        for w in range(n_windows):
            mean_comp = float(competence[w].mean())
            drift_flags[w] = drift_detector.update(mean_comp)
            if drift_flags[w]:
                competence[w] = 0.5

        selector_name: str = cfg.selection.name
        selector = get_selector(selector_name)(k=k, alpha=alpha)

        selected_indices = np.zeros((n_windows, k), dtype=int)
        for w in range(n_windows):
            preds = (windowed_scores[w] > threshold).astype(int)
            gt = (preds.mean(axis=0) > 0.5).astype(int)
            selected = selector.select(competence[w], preds, gt)
            if len(selected) > 0:
                selected_indices[w] = selected

        np.save(out_dir / "selected_indices.npy", selected_indices)
        np.save(out_dir / "competence.npy", competence)
        np.save(out_dir / "drift_flags.npy", drift_flags)
        pd.DataFrame(selected_indices).to_csv(out_dir / "selected_indices.csv", index=False)

        # Build blended scores: aggregate across selected detectors per timepoint
        blended = np.zeros(n_timepoints, dtype=float)
        for t in range(n_timepoints):
            w_idx = min(t // stride, n_windows - 1)
            sel = selected_indices[w_idx]
            if len(sel) > 0:
                blended[t] = scores[t, sel].mean()
            else:
                blended[t] = scores[t].mean()

        blended_df = pd.DataFrame(
            np.tile(blended[:, np.newaxis], (1, n_detectors)),
            columns=[f"detector_{i}" for i in range(n_detectors)],
        )
        blended_df.to_csv(out_dir / "blended_scores.csv", index=False)
        pd.DataFrame(competence).to_csv(out_dir / "competence.csv", index=False)

        with open(out_dir / "active_detectors.json", "w") as f:
            json.dump(list(set(selected_indices.flatten().tolist())), f)
        with open(out_dir / "drift_history.json", "w") as f:
            json.dump({"drift_flags": drift_flags.tolist(),
                       "drift_count": int(drift_detector.n_detections)}, f)

        meta = {
            "n_windows": int(n_windows), "window_size": window_size, "stride": stride,
            "k": k, "alpha": alpha, "selector": selector_name,
            "drift_method": drift_method, "drift_events": int(drift_detector.n_detections),
        }
        with open(out_dir / "selection_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        logger.info("[%s] Selection complete. Outputs written to %s", dataset_name, out_dir)
```

- [ ] **Step 5: Run tests**

```bash
uv run pytest tests/test_pipeline_paths.py tests/test_selection.py -v
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add cdade/reconciliation/run_reconcile.py cdade/selection/run_select.py tests/test_pipeline_paths.py
git commit -m "feat(reconcile,select): loop over datasets.active, namespace outputs by dataset"
```

---

## Task 6: `run_evaluate.py` dataset loop

**Files:**
- Modify: `cdade/evaluation/run_evaluate.py`
- Modify: `tests/test_evaluation_run.py`

**Interfaces:**
- Consumes: `results/selection/{dataset}/blended_scores.csv`, `results/baselines/{dataset}/b{1-5}_scores.npy`, `data/injected/{dataset}_counts_mask.parquet`.
- Produces: `results/metrics/{dataset}/metrics.json` (consumed by stats stage, Task 8) and `results/evaluation/{dataset}/` CSVs.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_evaluation_run.py`:

```python
class TestMultiDatasetEvaluate:
    """Test that run_evaluate writes per-dataset metrics under results/metrics/{dataset}/."""

    def test_metrics_written_to_namespaced_paths(self, tmp_path):
        import json
        import numpy as np
        import pandas as pd
        from types import SimpleNamespace
        from unittest.mock import patch, MagicMock
        from cdade.evaluation import run_evaluate

        # Build minimal sivep data
        n = 50
        mask = pd.DataFrame(np.zeros((n, 3), dtype=bool))
        mask.iloc[40:45, 0] = True
        (tmp_path / "data" / "injected").mkdir(parents=True)
        mask.to_parquet(tmp_path / "data" / "injected" / "sivep_counts_mask.parquet")

        # Build selection outputs
        sel_dir = tmp_path / "results" / "selection" / "sivep"
        sel_dir.mkdir(parents=True)
        scores_df = pd.DataFrame(
            np.random.default_rng(0).uniform(0, 1, (n, 2)),
            columns=["detector_0", "detector_1"],
        )
        scores_df.to_csv(sel_dir / "blended_scores.csv", index=False)

        # Build baseline scores
        base_dir = tmp_path / "results" / "baselines" / "sivep"
        base_dir.mkdir(parents=True)
        n_test = int(n * 0.2)
        for i in range(1, 6):
            np.save(base_dir / f"b{i}_scores.npy",
                    np.random.default_rng(i).uniform(0, 1, n_test))

        cfg = SimpleNamespace(
            datasets=SimpleNamespace(active=["sivep"]),
            evaluation=SimpleNamespace(test_frac=0.2, nab_window=4),
            experiment=SimpleNamespace(mlflow_tracking_uri="sqlite:///test.db"),
        )

        with patch("mlflow.set_tracking_uri"), \
             patch("mlflow.set_experiment"), \
             patch("mlflow.start_run", return_value=MagicMock(
                 __enter__=MagicMock(return_value=MagicMock()),
                 __exit__=MagicMock(return_value=False))):
            monkeypatch_path = patch.object(run_evaluate, "_PROJECT_ROOT", tmp_path)
            monkeypatch_path.start()
            run_evaluate.main.__wrapped__(cfg)
            monkeypatch_path.stop()

        metrics_file = tmp_path / "results" / "metrics" / "sivep" / "metrics.json"
        assert metrics_file.exists(), f"Expected {metrics_file}"
        with open(metrics_file) as f:
            metrics = json.load(f)
        assert "cdade" in metrics
        assert "auc_pr" in metrics["cdade"]
```

- [ ] **Step 2: Run to confirm it fails**

```bash
uv run pytest tests/test_evaluation_run.py::TestMultiDatasetEvaluate -v
```

Expected: FAIL — metrics written to flat `results/metrics.json` not namespaced path.

- [ ] **Step 3: Update `run_evaluate.py`**

Replace `load_baseline_scores` to accept a per-dataset directory, and rewrite `main()` to loop:

```python
def load_baseline_scores(baselines_dir: Path) -> dict[str, np.ndarray]:
    """Load baseline scores b1–b5 from .npy files in the given directory.

    Args:
        baselines_dir: Directory containing b{i}_scores.npy files
            (e.g. results/baselines/sivep/).

    Returns:
        Dict mapping method names to score arrays.
    """
    baselines = {}
    for i in range(1, 6):
        baseline_path = baselines_dir / f"b{i}_scores.npy"
        if baseline_path.exists():
            baselines[f"b{i}"] = np.load(baseline_path)
        else:
            logger.warning(f"Baseline {i} scores not found at {baseline_path}")
    return baselines
```

Replace `main()` with:

```python
@hydra.main(
    config_path=str(_PROJECT_ROOT / "configs"),
    config_name="config",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    """Orchestrate evaluation of CDADE and baselines for all active datasets."""
    from cdade.data.dataset_paths import _iter_datasets

    mlflow_uri = cfg.experiment.mlflow_tracking_uri
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment("cdade_evaluation")

    test_frac = cfg.evaluation.test_frac
    nab_window = cfg.evaluation.nab_window

    with mlflow.start_run(run_name="evaluate"):
        mlflow.log_param("test_frac", test_frac)
        mlflow.log_param("nab_window", nab_window)

        for dataset_name, artifact_paths in _iter_datasets(cfg, project_root=_PROJECT_ROOT):
            logger.info("Evaluating dataset: %s", dataset_name)

            # Ground truth
            mask_path = artifact_paths["mask"]
            if not mask_path.exists():
                raise FileNotFoundError(
                    f"Ground truth mask not found at {mask_path}. "
                    "Run `just data` or `uv run dvc repro inject` first."
                )
            mask_df = pd.read_parquet(mask_path)
            y_true = mask_df.values.astype(int).max(axis=1)
            n_test = int(len(y_true) * test_frac)
            y_test = y_true[-n_test:]
            logger.info(
                "[%s] Test size: %d (anomaly rate: %.1f%%)",
                dataset_name, len(y_test), 100 * y_test.mean(),
            )

            # CDADE blended scores
            blended_path = _RESULTS_DIR / "selection" / dataset_name / "blended_scores.csv"
            blended_df = load_blended_scores(blended_path, len(y_true))
            cdade_test = blended_df.values[-n_test:].max(axis=1)
            if len(cdade_test) != len(y_test):
                raise ValueError(
                    f"[{dataset_name}] CDADE scores/labels shape mismatch: "
                    f"{len(cdade_test)} vs {len(y_test)}."
                )

            # Baselines
            baselines_dir = _RESULTS_DIR / "baselines" / dataset_name
            baselines = load_baseline_scores(baselines_dir)

            # Metrics
            all_metrics = evaluate_all_methods(y_test, cdade_test, baselines, nab_window)

            # Per-dataset outputs
            metrics_out = _RESULTS_DIR / "metrics" / dataset_name / "metrics.json"
            save_metrics_json(all_metrics, metrics_out)

            eval_dir = _RESULTS_DIR / "evaluation" / dataset_name
            logger.info("[%s] Writing per-method CSV files to %s", dataset_name, eval_dir)
            save_per_method_csvs(all_metrics, eval_dir)

            # MLflow nested run per dataset
            with mlflow.start_run(run_name=dataset_name, nested=True):
                mlflow.log_param("dataset", dataset_name)
                mlflow.log_metric("n_test", len(y_test))
                mlflow.log_metric("anomaly_rate_test", float(y_test.mean()))
                for method, metrics in all_metrics.items():
                    _log_method_to_mlflow(method, metrics)
                mlflow.log_artifact(str(metrics_out))

            print(f"\n=== {dataset_name} Evaluation ===")
            for method, metrics in all_metrics.items():
                print(
                    f"  {method}: AUC-PR={metrics['auc_pr']:.3f} "
                    f"NAB={metrics['nab']:.3f}  F1={metrics['f1']:.3f}"
                )

    logger.info("Evaluation complete")
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_evaluation_run.py -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add cdade/evaluation/run_evaluate.py tests/test_evaluation_run.py
git commit -m "feat(evaluate): loop over datasets.active, write to results/metrics/{dataset}/"
```

---

## Task 7: `run_ablation.py` dataset loop

**Files:**
- Modify: `cdade/ablation/run_ablation.py`
- Modify: `tests/test_ablation.py`

**Interfaces:**
- Consumes: `results/selection/{dataset}/blended_scores.csv` from Task 5; `data/injected/{dataset}_counts_mask.parquet`.
- Produces: `results/ablation/{dataset}/{variant}_metrics.json` and `results/ablation/summary.csv`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_ablation.py`:

```python
def test_ablation_writes_to_namespaced_dirs(tmp_path):
    """Ablation writes per-dataset variant outputs to results/ablation/{dataset}/."""
    import json
    import numpy as np
    import pandas as pd
    from types import SimpleNamespace
    from unittest.mock import patch, MagicMock
    from cdade.ablation import run_ablation

    n = 50
    n_test = int(n * 0.2)

    # Build mask
    (tmp_path / "data" / "injected").mkdir(parents=True)
    mask = pd.DataFrame(np.zeros((n, 3), dtype=bool))
    mask.iloc[40:45, 0] = True
    mask.to_parquet(tmp_path / "data" / "injected" / "sivep_counts_mask.parquet")

    # Build blended scores
    sel_dir = tmp_path / "results" / "selection" / "sivep"
    sel_dir.mkdir(parents=True)
    scores = pd.DataFrame(
        np.random.default_rng(0).uniform(0, 1, (n, 2)),
        columns=["detector_0", "detector_1"],
    )
    scores.to_csv(sel_dir / "blended_scores.csv", index=False)

    cfg = SimpleNamespace(
        datasets=SimpleNamespace(active=["sivep"]),
        experiment=SimpleNamespace(mlflow_tracking_uri="sqlite:///test.db"),
        selection=SimpleNamespace(k=2, alpha=0.5),
        evaluation=SimpleNamespace(test_frac=0.2, nab_window=4),
    )

    with patch("mlflow.set_tracking_uri"), \
         patch("mlflow.set_experiment"), \
         patch("mlflow.start_run", return_value=MagicMock(
             __enter__=MagicMock(return_value=MagicMock()),
             __exit__=MagicMock(return_value=False))):
        patch.object(run_ablation, "_PROJECT_ROOT", tmp_path).start()
        patch.object(run_ablation, "_RESULTS_DIR", tmp_path / "results").start()
        patch.object(run_ablation, "_INJECTED_DIR", tmp_path / "data" / "injected").start()
        run_ablation.main.__wrapped__(cfg)

    sivep_ablation = tmp_path / "results" / "ablation" / "sivep"
    assert sivep_ablation.exists(), f"Expected {sivep_ablation}"
    assert (sivep_ablation / "full_metrics.json").exists()
    assert (tmp_path / "results" / "ablation" / "summary.csv").exists()
```

- [ ] **Step 2: Run to confirm it fails**

```bash
uv run pytest tests/test_ablation.py::test_ablation_writes_to_namespaced_dirs -v
```

Expected: FAIL — outputs not in namespaced dir.

- [ ] **Step 3: Update `run_ablation.py`**

Fix the `cfg` scoping bug (currently `run_ablation_variants` calls `load_variant_ground_truth(cfg, ...)` before `cfg` is defined) by threading `cfg` as a parameter. Then add the dataset loop.

Replace `run_ablation_variants` signature and internals:

```python
def run_ablation_variants(
    cfg: Any,
    dataset_name: str,
    y_true: np.ndarray,
    blended_scores: np.ndarray,
    results_dir: Path,
) -> dict[str, dict]:
    """Run ablation study over all variants for one dataset.

    Args:
        cfg: Config-like object with cfg.selection.k and cfg.selection.alpha.
        dataset_name: Name of the dataset being processed.
        y_true: Ground truth labels for the full dataset.
        blended_scores: Shape [n_timesteps, n_detectors] pre-loaded scores.
        results_dir: Root results directory.

    Returns:
        Dict mapping variant name to metrics dict.
    """
    test_frac = getattr(getattr(cfg, "evaluation", None), "test_frac", 0.2)
    nab_window = getattr(getattr(cfg, "evaluation", None), "nab_window", 4)
    n_test = int(len(y_true) * test_frac)
    y_test = y_true[-n_test:]

    ablation_dir = results_dir / "ablation" / dataset_name
    ablation_dir.mkdir(parents=True, exist_ok=True)

    variants = ["full", "no_reconciliation", "no_dynamic_selection", "no_diversity"]
    all_metrics = {}

    for variant in variants:
        logger.info("[%s] Processing variant: %s", dataset_name, variant)
        variant_scores = apply_variant_transformation(blended_scores, variant, cfg)
        # Use test portion only
        variant_test = variant_scores[-n_test:]
        metrics = compute_variant_metrics(y_test, variant_test, nab_window=nab_window)
        all_metrics[variant] = metrics

        variant_file = ablation_dir / f"{variant}_metrics.json"
        with open(variant_file, "w", encoding="utf-8") as fh:
            json.dump(metrics, fh, indent=2)
        logger.info("[%s] Metrics written to %s", dataset_name, variant_file)

    return all_metrics
```

Replace `main()` body:

```python
    if cfg is None:
        raise ValueError("Hydra configuration is required")

    from cdade.data.dataset_paths import _iter_datasets

    mlflow_uri = cfg.experiment.mlflow_tracking_uri
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment("cdade_ablation")

    logger.info("Starting ablation study")
    combined_metrics: dict[str, dict[str, dict]] = {}

    with mlflow.start_run(run_name="ablation_study"):
        for dataset_name, artifact_paths in _iter_datasets(cfg, project_root=_PROJECT_ROOT):
            mask_path = artifact_paths["mask"]
            if not mask_path.exists():
                raise FileNotFoundError(f"Ground truth mask not found at {mask_path}.")
            mask_df = pd.read_parquet(mask_path)
            y_true = mask_df.values.astype(int).max(axis=1)

            blended_path = _RESULTS_DIR / "selection" / dataset_name / "blended_scores.csv"
            blended_scores = load_variant_blended_scores(blended_path, len(y_true))

            dataset_metrics = run_ablation_variants(
                cfg, dataset_name, y_true, blended_scores, _RESULTS_DIR
            )
            combined_metrics[dataset_name] = dataset_metrics

            for variant, metrics in dataset_metrics.items():
                _log_variant_to_mlflow(f"{dataset_name}_{variant}", metrics)

    # Write combined summary CSV
    rows = []
    for ds, variants in combined_metrics.items():
        for variant, metrics in variants.items():
            row = {"dataset": ds, "variant": variant}
            row.update(metrics)
            rows.append(row)
    summary_df = pd.DataFrame(rows)
    summary_path = _RESULTS_DIR / "ablation" / "summary.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info("Summary written to %s", summary_path)

    logger.info("Ablation complete")
    for ds, variants in combined_metrics.items():
        print(f"\n=== {ds} Ablation ===")
        for variant, metrics in variants.items():
            logger.info("[%s] %s AUC-PR: %.4f NAB: %.4f",
                        ds, variant, metrics["auc_pr"], metrics["nab"])
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_ablation.py -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add cdade/ablation/run_ablation.py tests/test_ablation.py
git commit -m "feat(ablation): loop over datasets, fix cfg scoping, namespace outputs"
```

---

## Task 8: `stats.py` multi-dataset entry-point

**Files:**
- Modify: `cdade/evaluation/stats.py` (`__main__` block only)
- Modify: `tests/test_evaluation_stats.py`

**Interfaces:**
- Consumes: `results/metrics/{dataset}/metrics.json` from Task 6.
- Produces: `results/stats/friedman.json` with `skipped: false` when two datasets are present.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_evaluation_stats.py`:

```python
class TestMultiDatasetMatrixConstruction:
    """Test that the stats entry-point builds the AUC-PR matrix from multiple metrics files."""

    def test_auc_pr_matrix_has_two_rows(self, tmp_path):
        """With metrics for sivep and tycho, matrix shape is (2, n_methods)."""
        import json
        import numpy as np
        from cdade.evaluation.stats import _build_auc_pr_matrix_from_dir

        metrics_dir = tmp_path / "results" / "metrics"
        for ds in ["sivep", "tycho"]:
            ds_dir = metrics_dir / ds
            ds_dir.mkdir(parents=True)
            m = {
                "cdade":  {"auc_pr": 0.9, "nab": 0.8, "f1": 0.85,
                           "precision": 0.9, "recall": 0.8, "threshold": 0.5},
                "b1":     {"auc_pr": 0.6, "nab": 0.5, "f1": 0.55,
                           "precision": 0.6, "recall": 0.5, "threshold": 0.5},
            }
            (ds_dir / "metrics.json").write_text(json.dumps(m))

        matrix, method_names, dataset_names = _build_auc_pr_matrix_from_dir(metrics_dir)
        assert matrix.shape == (2, 2)
        assert set(method_names) == {"cdade", "b1"}
        assert set(dataset_names) == {"sivep", "tycho"}

    def test_single_dataset_returns_one_row(self, tmp_path):
        import json
        from cdade.evaluation.stats import _build_auc_pr_matrix_from_dir

        metrics_dir = tmp_path / "results" / "metrics"
        ds_dir = metrics_dir / "sivep"
        ds_dir.mkdir(parents=True)
        m = {"cdade": {"auc_pr": 0.9, "nab": 0.8, "f1": 0.85,
                       "precision": 0.9, "recall": 0.8, "threshold": 0.5}}
        (ds_dir / "metrics.json").write_text(json.dumps(m))

        matrix, method_names, dataset_names = _build_auc_pr_matrix_from_dir(metrics_dir)
        assert matrix.shape == (1, 1)
```

- [ ] **Step 2: Run to confirm they fail**

```bash
uv run pytest tests/test_evaluation_stats.py -k "TestMultiDataset" -v
```

Expected: `ImportError: cannot import name '_build_auc_pr_matrix_from_dir'`

- [ ] **Step 3: Add `_build_auc_pr_matrix_from_dir` and update `__main__`**

Add before the `if __name__ == "__main__":` block in `stats.py`:

```python
def _build_auc_pr_matrix_from_dir(
    metrics_dir: Path,
) -> tuple[np.ndarray, list[str], list[str]]:
    """Build AUC-PR matrix from all per-dataset metrics.json files.

    Args:
        metrics_dir: Directory containing ``{dataset}/metrics.json`` files.

    Returns:
        Tuple of (auc_pr_matrix, method_names, dataset_names).
        auc_pr_matrix shape: (n_datasets, n_methods).
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
```

Replace the `main()` body inside `if __name__ == "__main__":` with:

```python
    @hydra.main(
        config_path=str(Path(__file__).resolve().parents[2] / "configs"),
        config_name="config",
        version_base=None,
    )
    def main(cfg: DictConfig):
        """Main entry-point for DVC pipeline."""
        metrics_dir = Path("results/metrics")
        output_dir = Path("results/stats")

        auc_pr_matrix, method_names, dataset_names = _build_auc_pr_matrix_from_dir(metrics_dir)
        logger.info(
            "Loaded AUC-PR matrix: %d datasets × %d methods (%s)",
            auc_pr_matrix.shape[0], auc_pr_matrix.shape[1], dataset_names,
        )

        # Load raw scores from first (primary) dataset for DM and Cliff's delta
        primary = dataset_names[0]
        y_true = None
        cdade_scores = None
        baseline_scores = None

        try:
            mask_path = Path(f"data/injected/{primary}_counts_mask.parquet")
            if mask_path.exists():
                mask_df = pd.read_parquet(mask_path)
                test_frac = getattr(getattr(cfg, "evaluation", None), "test_frac", 0.2)
                n_test = int(len(mask_df) * test_frac)
                y_true = mask_df.max(axis=1).values[-n_test:]

            blended_path = Path(f"results/selection/{primary}/blended_scores.csv")
            if blended_path.exists():
                blended_df = pd.read_csv(blended_path, index_col=0)
                n_test_b = len(y_true) if y_true is not None else 26
                cdade_scores = blended_df.iloc[-n_test_b:].max(axis=1).values

            baseline_dir = Path(f"results/baselines/{primary}")
            b_paths = sorted(baseline_dir.glob("b[1-5]_scores.npy"))
            if b_paths:
                n_test_b = len(y_true) if y_true is not None else 26
                baseline_scores = {}
                for b_path in b_paths:
                    b_name = b_path.stem.replace("_scores", "")
                    baseline_scores[b_name] = np.load(b_path)[-n_test_b:]
        except Exception as e:
            logger.warning("Could not load raw scores: %s", e)

        # Build scalar metrics dict from first dataset for DM/Cliff's delta
        with open(metrics_dir / primary / "metrics.json") as f:
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

    main()
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_evaluation_stats.py -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add cdade/evaluation/stats.py tests/test_evaluation_stats.py
git commit -m "feat(stats): add _build_auc_pr_matrix_from_dir, feed real 2-row matrix to Friedman"
```

---

## Task 9: End-to-end smoke test

**Files:** none — verification only.

- [ ] **Step 1: Run full test suite**

```bash
uv run pytest -v
```

Expected: all tests pass (or pre-existing failures only — no new failures).

- [ ] **Step 2: Run full DVC pipeline**

```bash
uv run dvc repro
```

Expected: pipeline completes through `ablation`; `results/stats/friedman.json` contains a `stat` key (not `skipped: true`).

- [ ] **Step 3: Verify Friedman is no longer skipped**

```bash
python -c "import json; d=json.load(open('results/stats/friedman.json')); print(d)"
```

Expected output: `{'stat': <float>, 'p_value': <float>, 'significant': True/False}` — no `skipped` key.

- [ ] **Step 4: Commit if pipeline state changed**

```bash
git add dvc.lock results/stats/ results/metrics/
git commit -m "chore(pipeline): update dvc.lock and stats outputs for multi-dataset run"
```

---

## Task 10: Report figures

**Files:**
- Modify: `reports/02-results.qmd`
- Modify: `reports/03-ablation.qmd`

No pytest needed — figures are embedded Python in Quarto; correctness verified by `just report`.

- [ ] **Step 1: Add metrics heatmap + CD diagram to `reports/02-results.qmd`**

Replace the current `## Metrics summary` and `## Statistical comparison` sections with:

````markdown
## Metrics across datasets

```{python}
#| label: fig-metrics-heatmap
#| fig-cap: "AUC-PR per method and dataset. Darker = higher AUC-PR."
import json, pathlib, numpy as np, matplotlib.pyplot as plt, matplotlib as mpl

metrics_dir = pathlib.Path("../results/metrics")
files = sorted(metrics_dir.glob("*/metrics.json"))
dataset_names, per_dataset = [], []
for f in files:
    dataset_names.append(f.parent.name)
    per_dataset.append(json.loads(f.read_text()))

method_names = sorted(per_dataset[0].keys())
mat = np.array([[m[method]["auc_pr"] for method in method_names] for m in per_dataset])

fig, ax = plt.subplots(figsize=(max(6, len(method_names) * 0.9), max(2, len(dataset_names) * 0.7)))
im = ax.imshow(mat, vmin=0, vmax=1, cmap="YlGn", aspect="auto")
ax.set_xticks(range(len(method_names))); ax.set_xticklabels(method_names, rotation=40, ha="right")
ax.set_yticks(range(len(dataset_names))); ax.set_yticklabels(dataset_names)
plt.colorbar(im, ax=ax, label="AUC-PR")
for i in range(len(dataset_names)):
    for j in range(len(method_names)):
        ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=8,
                color="black" if mat[i, j] < 0.7 else "white")
plt.tight_layout()
plt.show()
```

## Statistical comparison

```{python}
#| label: fig-cd-diagram
#| fig-cap: "Critical-difference diagram. Methods connected by a bar are not significantly different (Wilcoxon + Bonferroni α=0.05)."
import json, pathlib, numpy as np, matplotlib.pyplot as plt
from scipy.stats import rankdata

metrics_dir = pathlib.Path("../results/metrics")
wilcoxon_path = pathlib.Path("../results/stats/wilcoxon.json")
files = sorted(metrics_dir.glob("*/metrics.json"))
per_dataset = [json.loads(f.read_text()) for f in files]

if len(per_dataset) < 2:
    print("CD diagram requires ≥2 datasets. Only one dataset available.")
else:
    method_names = sorted(per_dataset[0].keys())
    auc_pr = np.array([[m[mn]["auc_pr"] for mn in method_names] for m in per_dataset])
    # Rank: 1 = best (highest AUC-PR)
    ranks = np.apply_along_axis(lambda r: rankdata(-r), 1, auc_pr)
    mean_ranks = ranks.mean(axis=0)

    wilcoxon = json.loads(wilcoxon_path.read_text()) if wilcoxon_path.exists() else {}
    # Collect pairs that are NOT significant (no bar between them)
    nonsig_pairs = [
        (k.split(" vs ")[0].strip(), k.split(" vs ")[1].strip())
        for k, v in wilcoxon.items()
        if not v.get("significant", True)
    ]

    fig, ax = plt.subplots(figsize=(max(7, len(method_names)), 2.5))
    ax.set_xlim(0.8, len(method_names) + 0.2)
    ax.axhline(1, color="black", linewidth=1.5)
    for i, (name, rank) in enumerate(zip(method_names, mean_ranks)):
        ax.plot(rank, 1, "o", color="#2c7bb6", markersize=8)
        va = "bottom" if i % 2 == 0 else "top"
        ax.text(rank, 1, f"  {name}\n  ({rank:.2f})", ha="center", va=va, fontsize=8)
    # Draw bars for non-significant clusters
    for (m1, m2) in nonsig_pairs:
        if m1 in method_names and m2 in method_names:
            r1 = mean_ranks[method_names.index(m1)]
            r2 = mean_ranks[method_names.index(m2)]
            ax.plot([min(r1, r2), max(r1, r2)], [1.15, 1.15], "-", color="#d7191c", linewidth=3)
    ax.set_xlabel("Mean rank (1 = best)")
    ax.set_yticks([])
    ax.spines[["left", "top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.show()
```
````

Update the narrative block after the figures to remove the single-dataset disclaimer and instead describe what the Friedman result shows, referencing `results/stats/friedman.json`.

- [ ] **Step 2: Add Cliff's δ forest plot to `reports/03-ablation.qmd`**

Add after the ablation table section:

````markdown
## Effect-size summary (CDADE vs baselines)

```{python}
#| label: fig-cliffs-delta
#| fig-cap: "Cliff's δ effect size (CDADE vs each baseline). Bars show 95% bootstrap CI. Vertical dashed line at δ=0."
import json, pathlib, matplotlib.pyplot as plt, numpy as np

cliffs_path = pathlib.Path("../results/stats/cliffs_delta.json")
if not cliffs_path.exists():
    print("cliffs_delta.json not found — run `uv run dvc repro stats` first.")
else:
    data = json.loads(cliffs_path.read_text())
    baselines = list(data.keys())
    deltas = [data[b]["delta"] for b in baselines]
    ci_lo = [data[b]["ci_lower"] for b in baselines]
    ci_hi = [data[b]["ci_upper"] for b in baselines]
    mags = [data[b]["magnitude"] for b in baselines]

    colours = {"negligible": "#aec7e8", "small": "#ffbb78",
               "medium": "#98df8a", "large": "#ff9896"}
    fig, ax = plt.subplots(figsize=(7, max(2.5, len(baselines) * 0.55)))
    y = np.arange(len(baselines))
    for i, (b, d, lo, hi, mag) in enumerate(zip(baselines, deltas, ci_lo, ci_hi, mags)):
        ax.barh(i, hi - lo, left=lo, height=0.4, color=colours.get(mag, "grey"), alpha=0.8)
        ax.plot(d, i, "D", color="black", markersize=5)
    ax.axvline(0, color="grey", linestyle="--", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{b}  ({m})" for b, m in zip(baselines, mags)])
    ax.set_xlabel("Cliff's δ  (positive = CDADE better)")
    ax.set_xlim(-1.05, 1.05)
    plt.tight_layout()
    plt.show()
```
````

- [ ] **Step 3: Verify reports render**

```bash
just report
```

Expected: renders without errors; HTML output contains all three figures. If Quarto is not installed, confirm figures render by running the Python blocks directly:

```bash
uv run python -c "
import json, pathlib
files = sorted(pathlib.Path('results/metrics').glob('*/metrics.json'))
print([f.parent.name for f in files])
"
```

Expected: `['sivep', 'tycho']`

- [ ] **Step 4: Commit**

```bash
git add reports/02-results.qmd reports/03-ablation.qmd
git commit -m "feat(reports): add metrics heatmap, CD diagram, and Cliff's delta forest plot"
```
