import shutil
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

import cdade.registry as registry_module
from cdade.detectors import run_detect


class DummyDetector:
    def fit(self, data):
        return self

    def score(self, data):
        return pd.Series([0.5] * len(data), index=data.index)


def test_run_detect_writes_to_repo_results_dir(monkeypatch, tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = repo_root / "results" / "detectors"
    if output_dir.exists():
        shutil.rmtree(output_dir)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(registry_module, "get_detector", lambda name: DummyDetector())

    cfg = SimpleNamespace(detector=SimpleNamespace(name="dummy"))
    run_detect.run_detect(cfg)

    assert (output_dir / "leaf_forecasts.csv").exists()
    assert (output_dir / "detector_results.json").exists()

    shutil.rmtree(output_dir, ignore_errors=True)
