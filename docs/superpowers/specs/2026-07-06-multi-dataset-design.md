# Multi-Dataset Extension Design

**Date:** 2026-07-06  
**Status:** Approved  
**Scope:** Extend the CDADE pipeline from single-dataset (SIVEP) to multi-dataset (SIVEP + Tycho), enabling the full Friedman → Wilcoxon → DM → Cliff's δ hypothesis-testing protocol.

---

## Problem Statement

The current pipeline runs end-to-end but produces only one row in the AUC-PR matrix, so `stats.py` always skips the Friedman test and outputs placeholder statistics. Both raw datasets (SIVEP and Tycho) are already prepared and injected; the pipeline stages are simply not looping over them.

---

## Decisions

| Question | Decision | Rationale |
|----------|----------|-----------|
| DVC stage structure | **Loop inside stage (option B)** | Smallest diff; `dataset_paths.py` already provides namespacing |
| Test split per dataset | **Independent `test_frac=0.2`** | Each dataset uses its own natural test window |
| Report figures | **CD diagram + Cliff's δ forest plot + metrics heatmap** | Covers all README-specified outputs |

---

## Architecture

### Config additions

`params.yaml`:
```yaml
datasets:
  active: [sivep, tycho]
```

`configs/config.yaml` — no change needed; `datasets.active` is read from params.yaml.

### Output directory namespacing

All per-dataset results move from flat `results/{stage}/` to `results/{stage}/{dataset}/`:

```
results/
├── baselines/
│   ├── sivep/   b1_scores.npy … b5_scores.npy, metrics.json
│   └── tycho/   b1_scores.npy … b5_scores.npy, metrics.json
├── detectors/
│   ├── sivep/   leaf_forecasts.csv, detector_results.json
│   └── tycho/
├── reconciliation/
│   ├── sivep/
│   └── tycho/
├── selection/
│   ├── sivep/   blended_scores.csv
│   └── tycho/
├── evaluation/
│   ├── sivep/   cdade_metrics.csv, b1_metrics.csv …
│   └── tycho/
├── metrics/
│   ├── sivep/   metrics.json
│   └── tycho/   metrics.json
├── ablation/
│   ├── sivep/
│   └── tycho/
│   summary.csv        ← combined across datasets
└── stats/             ← unchanged; reads all metrics/*/metrics.json
    friedman.json, wilcoxon.json, diebold_mariano.json,
    cliffs_delta.json, summary.csv
```

The top-level `results/metrics.json` (current DVC metric) is replaced by `results/metrics/sivep/metrics.json` and `results/metrics/tycho/metrics.json`.

### `dvc.yaml` stage changes

Each affected stage gains:
- `params: [datasets.active]` as a dependency
- Updated `outs:` pointing to namespaced directories

```yaml
baselines:
  cmd: uv run python -m cdade.baselines.run_baselines
  deps: [cdade/baselines, configs, data/injected]
  params: [experiment.seed, datasets.active]
  metrics:
    - results/baselines_metrics.json:
        cache: false
  outs:
    - results/baselines

detect:
  deps: [cdade/detectors, cdade/registry.py, configs, data/injected]
  params: [datasets.active]
  outs:
    - results/detectors

reconcile:
  deps: [cdade/reconciliation, configs, results/detectors]
  params: [datasets.active]
  outs:
    - results/reconciliation

select:
  deps: [cdade/selection, configs, results/reconciliation]
  params: [datasets.active]
  outs:
    - results/selection

evaluate:
  deps: [cdade/evaluation, configs, results/selection, data/injected]
  params: [evaluation.test_frac, evaluation.nab_window, datasets.active]
  metrics:
    - results/metrics/sivep/metrics.json:
        cache: false
    - results/metrics/tycho/metrics.json:
        cache: false
  outs:
    - results/evaluation

stats:
  deps: [cdade/evaluation/stats.py, results/metrics]
  outs:
    - results/stats

ablation:
  deps: [cdade/ablation, cdade/evaluation, configs, results/selection, data/injected]
  params: [datasets.active]
  outs:
    - results/ablation
```

### Stage runner pattern

Each runner gains a shared `_iter_datasets(cfg)` generator (or inline loop):

```python
def _iter_datasets(cfg: DictConfig):
    active = list(cfg.datasets.active)
    for name in active:
        paths = get_dataset_artifact_paths(name, project_root=_PROJECT_ROOT)
        yield name, paths
```

Per-dataset outputs use `results/{stage}/{dataset}/` paths. Existing single-dataset logic is preserved inside the loop body — no changes to the actual detector/reconciler/selector calls.

### `stats.py` entry-point change

The `__main__` block replaces the hardcoded `sivep_counts_mask.parquet` path with a glob over `results/metrics/*/metrics.json`, stacking them into `(n_datasets, n_methods)`:

```python
metrics_files = sorted(Path("results/metrics").glob("*/metrics.json"))
# builds auc_pr_matrix shape (len(metrics_files), n_methods)
```

With two datasets the Friedman test proceeds normally.

---

## Report Changes

### `reports/02-results.qmd`

Three new embedded Python figures (no new dependencies beyond matplotlib which is already present via pyod/scipy):

1. **Metrics heatmap** — `imshow` with methods on y-axis, datasets on x-axis, AUC-PR values as cell colour. Reads `results/metrics/*/metrics.json`.

2. **CD diagram** — hand-rolled (~40 lines): horizontal number line, method names at their mean rank positions, a horizontal bar spanning the non-significant cluster (from `results/stats/friedman.json` + `wilcoxon.json`).

Updated narrative replaces the "single-dataset smoke test" disclaimer with actual Friedman result, Wilcoxon table, and interpretation.

### `reports/03-ablation.qmd`

3. **Cliff's δ forest plot** — `barh` with one row per CDADE-vs-baseline pair; horizontal bar = bootstrap CI, dot = point estimate. Reads `results/stats/cliffs_delta.json`.

Updated narrative includes DM statistics table and magnitude labels.

---

## Data Contracts

### Tycho test split

- Shape of `tycho_counts_injected.parquet`: `(n_tycho, n_cols)` (n_tycho determined at runtime)
- `test_frac = 0.2` → `n_test_tycho = int(n_tycho * 0.2)`
- `y_test_tycho = y_true_tycho[-n_test_tycho:]`
- Baseline scores for Tycho: `results/baselines/tycho/b{1-5}_scores.npy`, each shape `[n_test_tycho]`

### Friedman matrix

```
auc_pr_matrix[0, :] = [cdade, b1, b2, b3, b4, b5] AUC-PR on SIVEP test
auc_pr_matrix[1, :] = [cdade, b1, b2, b3, b4, b5] AUC-PR on Tycho test
```

Shape: `(2, 6)` → Friedman test proceeds (n_datasets ≥ 2 satisfied).

---

## What Is Not In Scope

- FluView (third dataset) — wiring deferred; the loop structure makes it trivial to add later by appending to `datasets.active`
- Changes to detector/reconciler/selector algorithm implementations
- New Python dependencies (matplotlib already present)

---

## Definition of Done

- [ ] `params.yaml` has `datasets.active: [sivep, tycho]`
- [ ] `dvc.yaml` stages updated with namespaced outs and `datasets.active` param dep
- [ ] Each runner loops over `datasets.active`, writes to `results/{stage}/{dataset}/`
- [ ] `stats.py` entry-point reads `results/metrics/*/metrics.json` and builds 2-row matrix
- [ ] `uv run dvc repro` runs cleanly end-to-end
- [ ] `results/stats/friedman.json` shows `significant: true/false` (not `skipped: true`)
- [ ] Reports render with all three figures populated from real data
- [ ] `uv run pytest` green
