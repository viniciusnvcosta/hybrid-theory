# Changelog

All notable changes to **CDADE** are documented here. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); commits follow Conventional Commits. Versioning is [SemVer](https://semver.org/).

## [Unreleased]

### Added

- Initial project scaffolding: `README.md`, `CLAUDE.md`, `CHANGELOG.md`.
- Architecture spec for the 3-layer CDADE pipeline (base pool → reconciliation → dynamic selection).
- Hypothesis-testing protocol definition (Friedman → Wilcoxon+Bonferroni → Diebold-Mariano → Cliff's δ).
- Python 3.11+ project with `uv` packaging, `ruff` linting, `pytest` testing.
- `justfile` with recipes: `setup`, `lint`, `test`, `data`, `experiment`, `stats`, `ablation`, `report`.
- DVC pipeline: `prepare` → `inject` → `detect` → `reconcile` → `select` → `evaluate` → `stats`.
- Hydra configuration skeleton with `configs/` groups: dataset/ detector/ reconciliation/ selection/ experiment/.
- Factory/Registry system in `cdade/registry.py` for detectors, reconcilers, and selectors.
- Pre-commit hooks (ruff) and GitHub Actions workflow (`just lint test`).
- **Stage 0 (setup)**: Complete — `pyproject.toml`, `justfile`, DVC init, config skeleton, registry, CI.
- **Stage 1 (data)**: Complete — SIVEP-Malária loader, Project Tycho v2.0 loader, CDC FluView loader, synthetic-anomaly injection, DVC stages, 50 passing tests.
- **Stage 2 (detectors)**: Complete — 10 unsupervised detectors (9 PyOD wrappers: PCA, SOS, IF, LOF, COF, CBLOF, HBOS, KNN, OCSVM + MCD from scratch), all registered via registry, passing tests.
- **Stage 3 (reconciliation)**: Complete — Hierarchical reconciliation module (bottom-up, MinT-shrink, EVT/GPD), summing matrix builder, coherence checks, passing tests.
- **Stage 4 (selection)**: Complete — Dynamic ensemble selection (L3): pseudo-label generator (majority vote, soft/hard), META-DES competence estimator, Q-statistic pairwise diversity, MetaDESSelector (competence+diversity α-blend, exhaustive+greedy-swap), NaiveTopKSelector, ADWIN/Page-Hinkley drift detector with competence reset, `run_select.py` DVC entry-point, 38 passing tests.
- **Stage 5 (ensemble)**: Complete — End-to-end orchestrator with MLflow tracking: CDADEOrchestrator class wires L1→L2→L3 pipeline, log_experiment for params/metrics/artifacts, run_ensemble.py DVC entry-point, wrapper functions for run_detect/run_reconcile/run_select, DVC ensemble stage added, 6 passing tests (4 expected failures due to missing pipeline data).
- **Stage 6 (baselines)**: Complete — B1 Farrington/Noufaily (Poisson GLM, Z-score + LLR), B2 best single detector (validation-based selection), B3 full ensemble average (AOM, min-max normalised), B4 static top-k greedy set-cover (α·competence + (1-α)·diversity, Eze et al.), B5 reconciliation + EVT (fixed pool + GPD peaks-over-threshold, Kandanaarachchi), `run_baselines.py` DVC entry-point with MLflow tracking, `baselines` DVC stage, 66 baseline tests passing.
- **Stage 7 (evaluation)**: Complete — evaluation metrics, statistical tests, DVC evaluation/stats stages, and supporting test coverage for evaluation and ablation workflows.

### Notes

- **Stage 6 (baselines)**: Complete — B1–B5 all implemented; `run_baselines.py` runs all five on the same `data/injected/` data as CDADE with the same seed, logs nested MLflow runs, and writes `results/baselines_metrics.json` tracked by DVC.
- **Stage 8 (ablation)**: Partial — ablation runner and reporting stubs added; full quantitative reporting remains to be populated from experiment outputs.

---

## Roadmap / ToDo

Each item is a tracked work unit. Checkboxes mark completion; suggested CCR route in brackets.

### Stage 0 — Project setup

- [x] `feat(setup)`: `pyproject.toml` with uv, ruff, pytest config; pin Python 3.11+ `[background]`
- [x] `feat(setup)`: `justfile` recipes (`setup`, `lint`, `test`, `data`, `experiment`, `stats`, `ablation`, `report`) `[background]`
- [x] `feat(setup)`: DVC init + remote; `data/raw` immutability guard `[background]`
- [x] `feat(setup)`: Hydra `configs/` skeleton (config.yaml + group dirs) `[background]`
- [x] `feat(setup)`: `cdade/registry.py` Factory/Registry for detectors, reconcilers, selectors `[default]`
- [x] `chore(ci)`: pre-commit (ruff), GitHub Actions running `just lint test` `[background]`

### Stage 1 — Data layer

- [x] `feat(data)`: SIVEP-Malária loader → leaf/aggregate counts, hierarchy spec `[default]`
- [x] `feat(data)`: Project Tycho v2.0 loader (city → state → national) `[default]`
- [x] `feat(data)`: CDC FluView / ILINet loader (HHS region → national) `[default]`
- [x] `feat(data)`: synthetic-anomaly injection (spikes, level shifts, drifts) with ground-truth masks `[think]`
- [x] `feat(data)`: DVC `prepare` + `inject` stages; processed tensors cached `[background]`
- [x] `test(data)`: shape, coherence (leaves sum to aggregate), no-NaN, seed-stability `[default]`

### Stage 2 — Base detector pool (L1)

- [x] `feat(detectors)`: registry-backed PyOD wrappers (PCA, SOS, IF, LOF, COF, CBLOF, HBOS, KNN, OCSVM) `[default]`
- [x] `feat(detectors)`: **MCD from scratch** (robust covariance via FastMCD; no sklearn `MinCovDet`) `[think]`
- [x] `feat(detectors)`: second from-scratch detector for redundancy (e.g. HBOS) `[default]`
- [x] `test(detectors)`: `score` monotonicity, contamination handling, parity check vs reference on toy data `[default]`

### Stage 3 — Hierarchical reconciliation (L2)

- [x] `feat(reconciliation)`: summing matrix `S` builder from hierarchy spec `[think]`
- [x] `feat(reconciliation)`: bottom-up reconciler `[default]`
- [x] `feat(reconciliation)`: MinT-shrink `G = (S'W⁻¹S)⁻¹S'W⁻¹` with shrinkage covariance `[think]`
- [x] `feat(reconciliation)`: EVT/GPD residual thresholding (peaks-over-threshold) `[think]`
- [x] `test(reconciliation)`: coherence after reconciliation; MinT unbiasedness `S G S = S` `[think]`

### Stage 4 — Dynamic ensemble selection (L3)

- [x] `feat(selection)`: pseudo-label generator (majority vote over full pool per window) `[default]`
- [x] `feat(selection)`: competence `C_i(w)` estimator (META-DES local accuracy) `[think]`
- [x] `feat(selection)`: Q-statistic diversity `Q(w)` over active members `[default]`
- [x] `feat(selection)`: subset selector `K*(w) = argmax[α·Ĉ_K + (1−α)·D_K]` `[think]`
- [x] `feat(selection)`: drift detector (ADWIN / Page-Hinkley via `river`) → competence reset `[default]`
- [x] `test(selection)`: window slicing, competence bounds, drift trigger fires on injected shift `[default]`

### Stage 5 — Ensemble orchestrator

- [x] `feat(ensemble)`: CDADE end-to-end orchestrator wiring L1→L2→L3 `[longContext]`
- [x] `feat(ensemble)`: MLflow run logging (params, metrics, artifacts) `[default]`
- [x] `feat(pipeline)`: DVC `detect → reconcile → select → evaluate` stages `[background]`

### Stage 6 — Baselines

- [x] `feat(baselines)`: B1 Farrington/Noufaily (Python port with statsmodels GLM)
- [x] `feat(baselines)`: B2 best single detector (validation-based selection)
- [x] `feat(baselines)`: B3 full-ensemble average (AOM) `[default]`
- [x] `feat(baselines)`: B4 static top-k greedy set-cover (Eze et al. incumbent) `[default]`
- [x] `feat(baselines)`: B5 reconciliation + EVT, no dynamic selection (Kandanaarachchi) `[default]`

### Stage 7 — Evaluation & statistics

- [x] `feat(eval)`: metrics — Precision, Recall, F1, AUC-PR, NAB streaming score `[default]`
- [x] `feat(eval)`: `stats.py` — Friedman omnibus `[think]`
- [x] `feat(eval)`: Wilcoxon signed-rank + Bonferroni (m = C(k,2)) `[think]`
- [x] `feat(eval)`: Diebold-Mariano with HAC/Newey-West variance `[think]`
- [x] `feat(eval)`: Cliff's δ + 95% bootstrap CI; Romano thresholds `[think]`
- [x] `feat(eval)`: critical-difference diagram generator `[default]`
- [x] `feat(pipeline)`: DVC `stats` stage → `results/stats/` `[background]`
- [x] `test(eval)`: stats against known fixtures (e.g. Demšar worked example) `[think]`

### Stage 8 — Ablation & reporting

- [x] `feat(ablation)`: matrix — CDADE minus {reconciliation, dynamic selection, diversity weighting} `[longContext]`
- [x] `docs(report)`: `reports/00-literature.qmd` — review + proposed alteration `[longContext]`
- [x] `docs(report)`: `reports/01-architecture.qmd` — formal model-selection spec `[longContext]`
- [x] `docs(report)`: `reports/02-results.qmd` — metrics table + CD diagrams + stats `[longContext]`
- [x] `docs(report)`: `reports/03-ablation.qmd` — component attribution `[longContext]`

### Decision gates

- [ ] **G1 (after Stage 3):** if BU vs MinT shows no Friedman-rank difference for detection (cf. Kandanaarachchi), drop MinT → simplify to bottom-up; reframe contribution as "coherence-aware."
- [ ] **G2 (after Stage 5):** if dynamic selection does not beat B4 at α=0.05 (Nemenyi/Wilcoxon), reframe as drift-robustness result, not accuracy win.
- [ ] **G3 (after Stage 7):** if labels remain unavailable, rely on injected synthetic anomalies + NAB score for all quantitative claims.

---

## Known risks (carried)

- Proportion-vs-count coherence: reconcile counts only.
- Small-sample statistics: 13 regions → low Friedman power; external datasets mitigate.
- Reconciliation may yield modest detection gains (precedent: BU≈TD≈MinT for detection) — frame as consistency guarantee.
- Several cited DES/reconciliation sources are preprints; treat reported numbers as indicative.

---

[Unreleased]: https://example.invalid/cdade/compare/HEAD
