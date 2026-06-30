"""Tests for CDADE ensemble orchestrator."""

from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import DictConfig

from cdade.ensemble.cdade import CDADEOrchestrator, run_ensemble


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

    def test_run_returns_dict(self, cfg: DictConfig):
        """Test run returns dictionary.

        Args:
            cfg: Test configuration
        """
        orchestrator = CDADEOrchestrator(cfg)
        results = orchestrator.run()
        assert isinstance(results, dict)
        assert "detectors" in results
        assert "reconciliation" in results
        assert "selection" in results

    def test_aggregate_metrics(self, cfg: DictConfig):
        """Test _aggregate_metrics generates expected keys.

        Args:
            cfg: Test configuration
        """
        orchestrator = CDADEOrchestrator(cfg)
        metrics = orchestrator._aggregate_metrics()
        assert isinstance(metrics, dict)
        # Should have at least these keys if pipeline ran
        assert "detector_count" in metrics or "reconciled_count" in metrics


class TestRunEnsemble:
    """Test run_ensemble convenience function."""

    def test_run_ensemble_returns_dict(self, cfg: DictConfig):
        """Test run_ensemble returns dictionary.

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

    def test_log_artifact(self, cfg: DictConfig):
        """Test log_artifact function.

        Args:
            cfg: Test configuration
        """
        from cdade.ensemble.logging import log_artifact

        data = {"key": "value", "list": [1, 2, 3]}
        log_artifact("test_artifact.json", data)

        # Verify file was created and deleted
        artifact_path = Path("test_artifact.json")
        assert artifact_path.exists()
        artifact_path.unlink()


class TestDVCStage:
    """Test DVC ensemble stage integration."""

    def test_run_ensemble_entry_point(self):
        """Test run_ensemble.py can be executed.

        This tests the DVC entry-point without actually running the full pipeline.
        """
        # Just verify the file can be imported
        from cdade.ensemble import run_ensemble

        assert callable(run_ensemble)

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
