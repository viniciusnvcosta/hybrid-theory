# CLAUDE.md — CDADE development instructions

Project instructions for Claude Code / Claude Scholar working on **CDADE** (Coherent Drift-Aware Dynamic Ensemble). Read this before touching code. Keep edits surgical and config-driven.

## Behavioral guidelines to reduce common LLM coding mistakes

> Tradeoff: These guidelines bias toward caution over speed. For trivial tasks, use judgment.

### 1. Think Before Coding

Don't assume. Don't hide confusion. Surface tradeoffs.

Before implementing:

State your assumptions explicitly. If uncertain, ask.
If multiple interpretations exist, present them - don't pick silently.
If a simpler approach exists, say so. Push back when warranted.
If something is unclear, stop. Name what's confusing. Ask.

### 2. Simplicity First

Minimum code that solves the problem. Nothing speculative.

No features beyond what was asked.
No abstractions for single-use code.
No "flexibility" or "configurability" that wasn't requested.
No error handling for impossible scenarios.
If you write 200 lines and it could be 50, rewrite it.
Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

### 3. Surgical Changes

Touch only what you must. Clean up only your own mess.

When editing existing code:

Don't "improve" adjacent code, comments, or formatting.
Don't refactor things that aren't broken.
Match existing style, even if you'd do it differently.
If you notice unrelated dead code, mention it - don't delete it.
When your changes create orphans:

Remove imports/variables/functions that YOUR changes made unused.
Don't remove pre-existing dead code unless asked.
The test: Every changed line should trace directly to the user's request.

### 4. Goal-Driven Execution

Define success criteria. Loop until verified.

Transform tasks into verifiable goals:

"Add validation" → "Write tests for invalid inputs, then make them pass"
"Fix the bug" → "Write a test that reproduces it, then make it pass"
"Refactor X" → "Ensure tests pass before and after"
For multi-step tasks, state a brief plan:

1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

These guidelines are working if: fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

---

## Project in one line

Hybrid anomaly detection for hierarchical time series that extends Eze et al. (2023) with hierarchical reconciliation (L2) and dynamic ensemble selection (L3), evaluated with a Friedman → Wilcoxon+Bonferroni → Diebold-Mariano → Cliff's δ protocol over 3 hierarchical datasets.

## Stack & invariants

- Python 3.11+ · **uv** (packaging) · **ruff** (lint+format) · **just** (tasks) · **DVC** (pipeline) · **Hydra+OmegaConf** (config) · **MLflow** (tracking) · **Quarto** (reports) · **pytest** (tests).
- **No sklearn-only solutions.** At least one base detector (MCD) is implemented from scratch in `cdade/detectors/`. sklearn may be a dependency but must not be the sole source of a method under comparison.
- **Raw data is immutable.** Never modify `data/raw/`. Write derived data to `data/processed/`, tracked by DVC.
- **Config over hard-coding.** Every path, hyperparameter, seed, and dataset choice lives in `configs/`. No literals in code paths.
- **Determinism.** Fixed seeds everywhere stochastic; record the seed per run.

## Coding style

- Type hints on all public functions; Google-style docstrings.
- Files 200–400 lines. Split when larger.
- Factory + Registry patterns for detectors, reconcilers, and selectors (`cdade/registry.py`). New components register by name; configs reference the name.
- Immutable config objects (frozen dataclasses / OmegaConf structured configs).
- Conventional Commits (`feat/fix/docs/refactor/test/chore`). Commit locally; do not push without explicit instruction.

## Layout (package mirrors tests)

```text
cdade/data · detectors · reconciliation · selection · ensemble · evaluation · registry.py
tests/     (mirrors the above, 1:1)
configs/   config.yaml + groups: dataset/ detector/ reconciliation/ selection/ experiment/
reports/   Quarto .qmd
dvc.yaml   prepare → inject → detect → reconcile → select → evaluate → stats
```

## Component contracts

- **Detector** (`cdade/detectors/`): `fit(X) -> self`, `score(X) -> np.ndarray` (higher = more anomalous). Registered via `@register_detector("name")`. MCD must not delegate to sklearn's `MinCovDet`.
- **Reconciler** (`cdade/reconciliation/`): builds summing matrix `S` from a hierarchy spec; exposes `reconcile(base_forecasts, W) -> coherent`. Implement `bottom_up`, `mint_shrink`. Reconcile on **counts**, never on proportions (coherence requires additivity — derive proportions post hoc).
- **Selector** (`cdade/selection/`): per-window competence `C_i(w)` from pseudo-labels, diversity `Q(w)`, returns active subset `K*(w)`. Drift detector triggers competence reset.
- **Evaluator** (`cdade/evaluation/`): metrics (AUC-PR, NAB, P/R/F1) + `stats.py` implementing the 4-stage protocol below.

## Hypothesis-testing protocol (do not reorder)

```text
1. Friedman omnibus on average ranks      → if p > .05, stop
2. Wilcoxon signed-rank, pairwise         → Bonferroni α/C(k,2)
3. Diebold-Mariano on predictive accuracy → HAC (Newey-West) variance
4. Cliff's delta effect size              → 95% bootstrap CI
```

Each region and each external dataset is one row in the Friedman table. Report ranks, corrected p-values, DM statistics, and δ with CI in `results/stats/`.

## Task routing (Claude Scholar / CCR)

- **background** — scaffolding, boilerplate, config stubs, docstring fills.
- **default** — module implementation (detectors, loaders, metrics).
- **think** — theory-critical: MinT derivation, EVT thresholding, META-DES competence math, DM/Cliff's δ correctness. Hard-assign these here.
- **longContext** — cross-module synthesis (orchestrator wiring, ablation matrix, report drafting).

## Do / Don't

- DO read `reports/01-architecture.qmd` before editing L2/L3 logic.
- DO add a pytest case mirroring every new module function before/with implementation.
- DO update `dvc.yaml` and `params.yaml` when adding a pipeline stage.
- DON'T introduce a new dependency without adding it to `pyproject.toml` via `uv add`.
- DON'T reconcile on proportions; reconcile counts.
- DON'T compare methods on point-adjusted F1 alone — AUC-PR + NAB are primary.
- DON'T push, change access controls, or delete data; ask the user.

## Definition of done (per component)

- [ ] Implemented with type hints + docstrings, ≤ 400 lines.
- [ ] Registered in the relevant registry; referenced by config, not import.
- [ ] Mirrored pre-commit + pytest passing; `just ci` green.
- [ ] DVC stage wired if it produces an artifact.
- [ ] MLflow logging in place for any metric it emits.
