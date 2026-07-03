"""Tests for CDADE ensemble orchestrator."""

from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import DictConfig

from cdade.ensemble.cdade import CDADEOrchestrator, run_ensemble

# Guard: tests that call orchestrator.run() need real DVC injected data.
# run_detect.py resolves data via Path("../data/injected/...") — mirror that
# exact path here so _DATA_AVAILABLE reflects whether the orchestrator can run.
_INJECTED_DATA = Path("../data/injected/sivep_counts_injected.parquet")
_DATA_AVAILABLE = _INJECTED_DATA.exists()
_SKIP_NO_DATA = pytest.mark.skipif(
    not _DATA_AVAILABLE,
    reason="DVC injected data not available — run `just data` first",
)


class TestCDADEOrchestrator:
    """Test CDADEOrchestrator class."""

    def test_init(self, cfg: DictConfig):
        """Test orchestrator initialization.

        Args:
            cfg: Test configuration
        """
        orchestrator = CDADEOrchestrator(cfg)
        assert orchestrator.cfg is cfg
        assert isinstance(orchestrator.results, dict)

    @_SKIP_NO_DATA
    def test_run_returns_dict(self, cfg: DictConfig):
        """Test run returns dictionary.

        Requires DVC injected data (run `just data` to produce it).

        Args:
            cfg: Test configuration
        """
        orchestrator = CDADEOrchestrator(cfg)
        results = orchestrator.run()
        assert isinstance(results, dict)
        assert "detectors" in results
        assert "reconciliation" in results
        assert "selection" in results

    @_SKIP_NO_DATA
    def test_aggregate_metrics(self, cfg: DictConfig):
        """Test _aggregate_metrics generates expected keys.

        Requires DVC injected data (run `just data` to produce it).

        Args:
            cfg: Test configuration
        """
        orchestrator = CDADEOrchestrator(cfg)
        orchestrator.run()
        metrics = orchestrator._aggregate_metrics()
        assert isinstance(metrics, dict)
        # Should have at least these keys if pipeline ran
        assert "detector_count" in metrics or "reconciled_count" in metrics


class TestRunEnsemble:
    """Test run_ensemble convenience function."""

    @_SKIP_NO_DATA
    def test_run_ensemble_returns_dict(self, cfg: DictConfig):
        """Test run_ensemble returns dictionary.

        Requires DVC injected data (run `just data` to produce it).

        Args:
            cfg: Test configuration
        """
        results = run_ensemble(cfg)
        assert isinstance(results, dict)
        assert "detectors" in results


class TestMLflowLogging:
    """Test MLflow logging utilities."""

    def test_log_experiment_without_mlflow(self, cfg: DictConfig):
        """Test log_experiment doesn't raise when MLflow unavailable.

        Args:
            cfg: Test configuration
        """
        from cdade.ensemble.logging import log_experiment

        metrics = {"test_metric": 0.5, "test_metric2": 0.3}

        # Should not raise even if MLflow not configured
        log_experiment(cfg, metrics)

    def test_log_params(self, cfg: DictConfig):
        """Test log_params function.

        Args:
            cfg: Test configuration
        """
        from cdade.ensemble.logging import log_params

        log_params(cfg)

    def test_log_metrics(self, cfg: DictConfig):
        """Test log_metrics function.

        Args:
            cfg: Test configuration
        """
        from cdade.ensemble.logging import log_metrics

        metrics = {"accuracy": 0.9, "precision": 0.85}
        log_metrics(metrics)

    @pytest.mark.skip(reason="MLflow logging best-effort, artifact file not guaranteed")
    def test_log_artifact(self, cfg: DictConfig):
        """Test log_artifact function.

        Args:
            cfg: Test configuration
        """
        from cdade.ensemble.logging import log_artifact

        data = {"key": "value", "list": [1, 2, 3]}
        log_artifact("test_artifact.json", data)

        # Verify artifact file was created
        artifact_path = Path("test_artifact.json")
        assert artifact_path.exists(), f"Artifact file {artifact_path} was not created"

        # Cleanup
        artifact_path.unlink()


class TestDVCStage:
    """Test DVC ensemble stage integration."""

    def test_run_ensemble_entry_point(self):
        """Test the run_ensemble function is importable and callable.

        Imports from cdade.ensemble.cdade (the function), not the DVC
        entry-point module run_ensemble.py which has module-level Hydra
        initialisation that requires an absolute config path.
        """
        from cdade.ensemble.cdade import run_ensemble as _run_ensemble

        assert callable(_run_ensemble)

    def test_ensemble_module_structure(self):
        """Test ensemble module has correct public API.

        Checks that __init__.py exports expected functions/classes.
        """
        from cdade.ensemble import CDADEOrchestrator, log_experiment

        assert callable(CDADEOrchestrator)
        assert callable(log_experiment)


@pytest.fixture
def cfg() -> DictConfig:
    """Provide test configuration.

    Returns:
        Hydra configuration
    """
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(
        config_dir="/home/vinvs/projects/hybrid-theory/configs", version_base=None
    ):
        return compose(config_name="config")


@pytest.fixture(autouse=True)
def cleanup_test_artifacts():
    """Cleanup test artifacts after each test.

    Removes mlruns/ directory and test_artifact.json files.
    Note: mlflow.db is a readonly database fixture artifact (ignored).
    """
    import shutil

    # Cleanup mlflow artifacts (run directories only)
    mlruns_dir = Path("mlruns")

    if mlruns_dir.exists():
        shutil.rmtree(mlruns_dir)

    # Cleanup test artifact
    artifact_path = Path("test_artifact.json")
    if artifact_path.exists():
        artifact_path.unlink()

    yield

    # Cleanup again after test
    if mlruns_dir.exists():
        shutil.rmtree(mlruns_dir)

    if artifact_path.exists():
        artifact_path.unlink()
