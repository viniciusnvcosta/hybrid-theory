# CDADE task runner. Keep recipes small and explicit.
set shell := ["bash", "-euo", "pipefail", "-c"]

package := "cdade"
paths := "cdade tests"

_default:
    @just --list

# Stage-0 setup: no PyOD/numba detector stack yet.
setup:
    uv sync --group dev --group workflow
    @if [ -f .pre-commit-config.yaml ]; then uv run pre-commit install; fi
    @if [ -d .dvc ]; then uv run dvc pull --allow-missing || true; fi
    just raw-guard

# Full research environment for Stage 1+ experiments.
setup-experiment:
    uv sync --group dev --group workflow --group data --group experiment
    @if [ -f .pre-commit-config.yaml ]; then uv run pre-commit install; fi
    @if [ -d .dvc ]; then uv run dvc pull --allow-missing || true; fi
    just raw-guard

lock:
    uv lock

lock-upgrade:
    uv lock --upgrade

# Useful after changing dependency bounds that affect a bad lock entry.
lock-upgrade-numba:
    uv lock --upgrade-package numba --upgrade-package pyod

doctor:
    uv run python -c "import sys, numpy, pandas, scipy, hydra; print(sys.version); print('core imports ok')"

lint:
    uv run ruff check {{paths}}
    uv run ruff format --check {{paths}}

fmt:
    uv run ruff check --fix {{paths}}
    uv run ruff format {{paths}}

test *args:
    uv run pytest {{args}}

cov:
    uv run pytest --cov={{package}} --cov-report=term-missing

# Guardrail: DVC stages must not produce data/raw outputs.
raw-guard:
    @if [ -f dvc.yaml ] && grep -nE '^[[:space:]]*-[[:space:]]+data/raw' dvc.yaml; then \
        echo 'ERROR: data/raw must be an input/dependency, not a DVC stage output.'; \
        exit 1; \
    else \
        echo 'raw guard ok'; \
    fi

data stage="prepare_fluview":
    uv run dvc repro {{stage}}

repro:
    uv run dvc repro

experiment name="cdade_full":
    uv run python -m cdade.ensemble.run experiment={{name}}

stats:
    uv run python -m cdade.evaluation.stats

ablation:
    uv run python -m cdade.evaluation.ablation

report:
    @command -v quarto >/dev/null || { echo 'Quarto CLI not found. Install Quarto first.'; exit 1; }
    quarto render reports --output-dir ../results/reports
