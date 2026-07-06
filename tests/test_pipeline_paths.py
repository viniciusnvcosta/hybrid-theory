# ABOUTME: Tests that pipeline stage runners write outputs to correct paths.
# ABOUTME: Verifies namespaced dataset output directories.

from types import SimpleNamespace

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

    run_detect.run_detect(cfg, dataset_name="sivep")

    out_dir = tmp_path / "results" / "detectors" / "sivep"
    assert (out_dir / "leaf_forecasts.csv").exists()
    assert (out_dir / "detector_results.json").exists()
