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
- **Stage 2 (detectors)**: Pending — 10 unsupervised detectors (9 PyOD wrappers + MCD from scratch) not yet implemented.
- **Stage 3 (reconciliation)**: Pending — Hierarchical reconciliation module (bottom-up, MinT-shrink, EVT) not yet implemented.

### Notes

- Stages 2–8 pending — see roadmap below.

---

## Roadmap / ToDo

Each item is a tracked work unit. Checkboxes mark completion; suggested CCR route in brackets.

### Stage 0 — Project setup

- [ ] `feat(setup)`: `pyproject.toml` with uv, ruff, pytest config; pin Python 3.11+ `[background]`
- [ ] `feat(setup)`: `justfile` recipes (`setup`, `lint`, `test`, `data`, `experiment`, `stats`, `ablation`, `report`) `[background]`
- [ ] `feat(setup)`: DVC init + remote; `data/raw` immutability guard `[background]`
- [ ] `feat(setup)`: Hydra `configs/` skeleton (config.yaml + group dirs) `[background]`
- [ ] `feat(setup)`: `cdade/registry.py` Factory/Registry for detectors, reconcilers, selectors `[default]`
- [ ] `chore(ci)`: pre-commit (ruff), GitHub Actions running `just lint test` `[background]`

### Stage 1 — Data layer

- [ ] `feat(data)`: SIVEP-Malária loader → leaf/aggregate counts, hierarchy spec `[default]`
- [ ] `feat(data)`: Project Tycho v2.0 loader (city → state → national) `[default]`
- [ ] `feat(data)`: CDC FluView / ILINet loader (HHS region → national) `[default]`
- [ ] `feat(data)`: synthetic-anomaly injection (spikes, level shifts, drifts) with ground-truth masks `[think]`
- [ ] `feat(data)`: DVC `prepare` + `inject` stages; processed tensors cached `[background]`
- [ ] `test(data)`: shape, coherence (leaves sum to aggregate), no-NaN, seed-stability `[default]`

### Stage 2 — Base detector pool (L1)

- [ ] `feat(detectors)`: registry-backed PyOD wrappers (PCA, SOS, IF, LOF, COF, CBLOF, HBOS, KNN, OCSVM) `[default]`
- [ ] `feat(detectors)`: **MCD from scratch** (robust covariance via FastMCD; no sklearn `MinCovDet`) `[think]`
- [ ] `feat(detectors)`: second from-scratch detector for redundancy (e.g. HBOS) `[default]`
- [ ] `test(detectors)`: `score` monotonicity, contamination handling, parity check vs reference on toy data `[default]`

### Stage 3 — Hierarchical reconciliation (L2)

- [ ] `feat(reconciliation)`: summing matrix `S` builder from hierarchy spec `[think]`
- [ ] `feat(reconciliation)`: bottom-up reconciler `[default]`
- [ ] `feat(reconciliation)`: MinT-shrink `G = (S'W⁻¹S)⁻¹S'W⁻¹` with shrinkage covariance `[think]`
- [ ] `feat(reconciliation)`: EVT/GPD residual thresholding (peaks-over-threshold) `[think]`
- [ ] `test(reconciliation)`: coherence after reconciliation; MinT unbiasedness `S G S = S` `[think]`

### Stage 4 — Dynamic ensemble selection (L3)

- [ ] `feat(selection)`: pseudo-label generator (majority vote over full pool per window) `[default]`
- [ ] `feat(selection)`: competence `C_i(w)` estimator (META-DES local accuracy) `[think]`
- [ ] `feat(selection)`: Q-statistic diversity `Q(w)` over active members `[default]`
- [ ] `feat(selection)`: subset selector `K*(w) = argmax[α·Ĉ_K + (1−α)·D_K]` `[think]`
- [ ] `feat(selection)`: drift detector (ADWIN / Page-Hinkley via `river`) → competence reset `[default]`
- [ ] `test(selection)`: window slicing, competence bounds, drift trigger fires on injected shift `[default]`

### Stage 5 — Ensemble orchestrator

- [ ] `feat(ensemble)`: CDADE end-to-end orchestrator wiring L1→L2→L3 `[longContext]`
- [ ] `feat(ensemble)`: MLflow run logging (params, metrics, artifacts) `[default]`
- [ ] `feat(pipeline)`: DVC `detect → reconcile → select → evaluate` stages `[background]`

### Stage 6 — Baselines

- [ ] `feat(baselines)`: B1 Farrington/Noufaily (call R `surveillance` or port) `[think]`
- [ ] `feat(baselines)`: B2 best single detector; B3 full-ensemble average (AOM) `[default]`
- [ ] `feat(baselines)`: B4 static top-k greedy set-cover (Eze et al. incumbent) `[default]`
- [ ] `feat(baselines)`: B5 reconciliation + EVT, no dynamic selection (Kandanaarachchi) `[default]`

### Stage 7 — Evaluation & statistics

- [ ] `feat(eval)`: metrics — Precision, Recall, F1, AUC-PR, NAB streaming score `[default]`
- [ ] `feat(eval)`: `stats.py` — Friedman omnibus `[think]`
- [ ] `feat(eval)`: Wilcoxon signed-rank + Bonferroni (m = C(k,2)) `[think]`
- [ ] `feat(eval)`: Diebold-Mariano with HAC/Newey-West variance `[think]`
- [ ] `feat(eval)`: Cliff's δ + 95% bootstrap CI; Romano thresholds `[think]`
- [ ] `feat(eval)`: critical-difference diagram generator `[default]`
- [ ] `feat(pipeline)`: DVC `stats` stage → `results/stats/` `[background]`
- [ ] `test(eval)`: stats against known fixtures (e.g. Demšar worked example) `[think]`

### Stage 8 — Ablation & reporting

- [ ] `feat(ablation)`: matrix — CDADE minus {reconciliation, dynamic selection, diversity weighting} `[longContext]`
- [ ] `docs(report)`: `reports/00-literature.qmd` — review + proposed alteration `[longContext]`
- [ ] `docs(report)`: `reports/01-architecture.qmd` — formal model-selection spec `[longContext]`
- [ ] `docs(report)`: `reports/02-results.qmd` — metrics table + CD diagrams + stats `[longContext]`
- [ ] `docs(report)`: `reports/03-ablation.qmd` — component attribution `[longContext]`

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
