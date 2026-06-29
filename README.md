# CDADE — Coherent Drift-Aware Dynamic Ensemble

> Hybrid anomaly detection for **hierarchical time series**, extending the static top-k ensemble of Eze et al. (2023) with hierarchical reconciliation, dynamic ensemble selection, and temporal diversity weighting.

**Course:** Reconhecimento de Padrões (Machine Learning) — POLI-UPE
**Status:** scaffolding · Stage 0 (project setup)

---

## 1. Motivation

Eze et al. (2023, _Healthcare_ 11(13):1896) detect anomalies in hierarchical malaria-surveillance series with an ensemble of 10 unsupervised detectors (PCA, SOS, MCD, Isolation Forest, LOF, COF, CBLOF, HBOS, KNN, OCSVM) and a **greedy set-cover** to pick a static top-k subset. The heuristic carries a `(1 − 1/e)` approximation guarantee but optimizes the wrong objective for hierarchical time series. CDADE targets four documented limitations:

| Limitation              | Eze et al. (2023)               | CDADE                                                     |
| ----------------------- | ------------------------------- | --------------------------------------------------------- |
| **Static selection**    | top-k fixed over full record    | online selection per sliding window                       |
| **Coherence-blind**     | regions treated independently   | reconciliation across leaf → aggregate (MinT / bottom-up) |
| **Temporally agnostic** | point-based, i.i.d. scoring     | windowed competence + drift detection                     |
| **Label circularity**   | covers a model-defined universe | pseudo-label competence + injected/synthetic ground truth |

**Central thesis:** replacing static greedy top-k with (1) coherence reconciliation and (2) competence-driven dynamic selection yields a statistically superior pipeline under a rigorous multi-dataset hypothesis-testing protocol.

---

## 2. Architecture

```text
Raw counts ──▶ [L1] Base detector pool ──▶ [L2] Hierarchical ──▶ [L3] Dynamic ──▶ Reconciled
              (≥1 from-scratch, e.g. MCD)     reconciliation      selection        anomaly score
                                              (S, MinT, EVT)      (META-DES + Q)
```

- **L1 — Base detector pool.** 10 unsupervised detectors. At least one (MCD) is implemented from scratch (numpy/scipy) to satisfy the "no sklearn-only" requirement.
- **L2 — Hierarchical reconciliation.** Summing matrix `S`, MinT-shrink `G = (S'W⁻¹S)⁻¹S'W⁻¹` (and bottom-up baseline), residual-space detection with EVT/GPD thresholding (Kandanaarachchi et al., 2023 template).
- **L3 — Dynamic ensemble selection.** Per-window competence via pseudo-labels (META-DES), Q-statistic diversity weighting, drift trigger (ADWIN / Page-Hinkley) → pool reset.

See `reports/01-architecture.qmd` for the formal specification.

---

## 3. Tech stack

| Concern                    | Tool                          |
| -------------------------- | ----------------------------- |
| Language                   | Python 3.11+                  |
| Packaging / venv           | **uv**                        |
| Lint + format              | **ruff**                      |
| Task runner                | **just**                      |
| Pipeline / data versioning | **DVC**                       |
| Config                     | **Hydra + OmegaConf**         |
| Experiment tracking        | **MLflow**                    |
| Reports / paper            | **Quarto**                    |
| Tests                      | **pytest** (mirrors `cdade/`) |

External (non-sklearn) deps: `pyod`, `statsmodels` (Diebold-Mariano), `scipy`, `river` (drift detectors), `numpy`, `pandas`.

---

## 4. Repository layout

```text
cdade/
├── README.md · CLAUDE.md · CHANGELOG.md
├── pyproject.toml · justfile · dvc.yaml · params.yaml
├── cdade/                      # package (mirrors tests/)
│   ├── data/                   # loaders: sivep, tycho, fluview
│   ├── detectors/              # base detectors + registry (MCD from scratch)
│   ├── reconciliation/         # summing matrix, MinT, bottom-up, EVT
│   ├── selection/              # competence, diversity (Q-stat), META-DES, drift
│   ├── ensemble/               # CDADE orchestrator
│   ├── evaluation/             # metrics (AUC-PR, NAB) + stats protocol
│   └── registry.py             # Factory / Registry
├── configs/                    # Hydra group configs
│   ├── config.yaml
│   ├── dataset/ · detector/ · reconciliation/ · selection/ · experiment/
├── tests/                      # pytest suite (mirrors cdade/)
├── reports/                    # Quarto .qmd (architecture, results, ablation)
├── notebooks/                  # 01-* exploration
├── data/{raw,processed}/       # DVC-tracked (raw is immutable)
└── results/                    # experiment outputs, figures, stats tables
```

---

## 5. Datasets (all hierarchical)

| Dataset              | Hierarchy               | Series | Period    | Access                      |
| -------------------- | ----------------------- | ------ | --------- | --------------------------- |
| SIVEP-Malária (Pará) | região → estado         | 13 + 1 | 2009–2019 | public (Baroni et al. 2020) |
| Project Tycho v2.0   | city → state → national | 1,284+ | 1888–2017 | public (DOI-indexed)        |
| CDC FluView / ILINet | HHS region → national   | 10 + 1 | 2010–2023 | public API                  |

Raw data is immutable under `data/raw/` and tracked by DVC. Synthetic-anomaly injection (controlled spikes/drifts) provides quantitative ground truth where labels are absent.

---

## 6. Baselines & evaluation

**Baselines (increasing sophistication):**

1. `B1` Farrington / Noufaily (R `surveillance`, per region) — public-health standard
2. `B2` best single PyOD detector
3. `B3` full-ensemble average (AOM)
4. `B4` **static top-k** (Eze et al. 2023) — direct incumbent
5. `B5` reconciliation + EVT (Kandanaarachchi 2023, no dynamic selection)
6. `P` **CDADE** (full)

**Metrics:** Precision, Recall, F1, AUC-PR, NAB streaming score.

**Hypothesis-testing protocol** (`cdade/evaluation/stats.py`):

```text
[1] Friedman omnibus            (p ≤ 0.05 ?)  ── reject → continue
[2] Wilcoxon signed-rank        + Bonferroni (m = C(k,2)) pairwise
[3] Diebold-Mariano             predictive accuracy, HAC/Newey-West
[4] Cliff's Delta               effect size + 95% bootstrap CI
```

Each region and external dataset counts as one "dataset" in the Friedman ranking.

---

## 7. Quickstart

```bash
# 0. install uv (once):  curl -LsSf https://astral.sh/uv/install.sh | sh
just setup            # uv sync + dvc pull + pre-commit install
just lint             # ruff check + format --check
just test             # pytest
just repro            # dvc repro — full pipeline (data → detect → reconcile → select → eval)
just report           # quarto render reports/
```

Common recipes (see `justfile`):

| Recipe                            | Action                                      |
| --------------------------------- | ------------------------------------------- |
| `just setup`                      | sync env, pull data, install hooks          |
| `just data`                       | `dvc repro` data-prep stages only           |
| `just experiment name=cdade_full` | run one Hydra experiment                    |
| `just stats`                      | run the 4-stage hypothesis-testing protocol |
| `just ablation`                   | run component ablation matrix               |
| `just report`                     | render Quarto reports to `results/reports/` |

---

## 8. Reproducibility

- Fixed seeds across all stochastic components; seed recorded per run.
- Every experiment is a Hydra config under `configs/experiment/`; no hard-coded paths.
- `dvc.yaml` defines the DAG: `prepare → inject → detect → reconcile → select → evaluate → stats`.
- MLflow logs params, metrics, and artifacts per run.
- Quarto reports pin the exact run IDs they summarize.

---

## 9. Course requirements traceability

| Requirement                             | Where satisfied                                                 |
| --------------------------------------- | --------------------------------------------------------------- |
| Hybrid system                           | CDADE 3-layer ensemble (§2)                                     |
| Literature review + proposed alteration | `reports/00-literature.qmd`; reconciliation + DES (§1–2)        |
| Comparative experiments vs baselines    | §6, `dvc.yaml` `evaluate` stage                                 |
| Not sklearn-only                        | MCD (and ≥1 more) implemented from scratch (`cdade/detectors/`) |
| Mandatory hypothesis testing            | §6 4-stage protocol (`cdade/evaluation/stats.py`)               |
| More than one dataset                   | §5 — 3 hierarchical datasets                                    |
| Metrics table                           | §6 / `results/metrics.csv`                                      |
| Model-selection process spec            | §2 L3 + `reports/01-architecture.qmd`                           |

---

## 10. References (anchor)

- Eze, Geard, Mueller & Chadès (2023). _Healthcare_ 11(13):1896.
- Wickramasuriya, Athanasopoulos & Hyndman (2019). MinT. _JASA_ 114(526).
- Kandanaarachchi et al. (2023). Hierarchical prediction + EVT for anomalous nodes. arXiv:2304.13941.
- Cruz, Sabourin, Cavalcanti & Ren (2015). META-DES. _Pattern Recognition_.
- Kuncheva & Whitaker (2003). Diversity measures. _Machine Learning_.
- Demšar (2006). Statistical comparisons of classifiers. _JMLR_.
- Diebold & Mariano (1995). Comparing predictive accuracy. _J. Bus. Econ. Stat._
- Romano et al. (2006). Cliff's delta thresholds.

---

_Licensed MIT. See `CHANGELOG.md` for the roadmap and `CLAUDE.md` for development conventions._
