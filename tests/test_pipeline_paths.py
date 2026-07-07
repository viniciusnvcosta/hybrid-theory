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


def test_run_reconcile_writes_to_namespaced_dir(monkeypatch, tmp_path):
    """reconcile stage writes outputs to results/reconciliation/{dataset}/."""
    import json

    import numpy as np
    import pandas as pd

    from cdade.reconciliation import run_reconcile as rr_module

    # Create required input: hierarchy json and detector outputs
    hierarchy = {"leaves": ["a", "b"], "aggregate": "total", "S": [[1, 0], [0, 1], [1, 1]]}
    det_dir = tmp_path / "results" / "detectors" / "sivep"
    det_dir.mkdir(parents=True)
    processed_dir = tmp_path / "data" / "processed"
    processed_dir.mkdir(parents=True)
    (processed_dir / "hierarchy_sivep.json").write_text(json.dumps(hierarchy))

    scores_df = pd.DataFrame(np.random.default_rng(0).uniform(0, 1, (10, 2)), columns=["a", "b"])
    scores_df.to_csv(det_dir / "leaf_forecasts.csv", index=False)

    monkeypatch.setattr(rr_module, "_PROJECT_ROOT", tmp_path)

    cfg = SimpleNamespace(
        datasets=SimpleNamespace(active=["sivep"]),
        reconciliation=SimpleNamespace(name="bottom_up"),
        dataset=SimpleNamespace(name="sivep"),
    )

    rr_module.run_reconcile(cfg, dataset_name="sivep")

    # Verify the output dir is namespaced
    out_dir = tmp_path / "results" / "reconciliation" / "sivep"
    assert (out_dir / "leaf_forecasts_reconciled.csv").exists()
    assert (out_dir / "aggregate_forecasts.csv").exists()
    assert (out_dir / "hierarchy.json").exists()


def test_run_select_reads_from_namespaced_reconcile_dir(monkeypatch, tmp_path):
    """select stage reads leaf_forecasts_reconciled.csv from results/reconciliation/{dataset}/."""
    import numpy as np
    import pandas as pd

    from cdade.selection import run_select as rs_module

    recon_dir = tmp_path / "results" / "reconciliation" / "sivep"
    recon_dir.mkdir(parents=True)
    df = pd.DataFrame(np.random.default_rng(0).uniform(0, 1, (20, 3)))
    df.to_csv(recon_dir / "leaf_forecasts_reconciled.csv")

    monkeypatch.setattr(rs_module, "_PROJECT_ROOT", tmp_path)

    cfg = SimpleNamespace(
        datasets=SimpleNamespace(active=["sivep"]),
        selection=SimpleNamespace(
            name="meta_des",
            window=5,
            stride=1,
            alpha=0.5,
            k=2,
            threshold=0.5,
            drift_method="adwin",
        ),
    )
    # Run the main function (it should loop over datasets)
    rs_module.main.__wrapped__(cfg)

    out_dir = tmp_path / "results" / "selection" / "sivep"
    assert (out_dir / "blended_scores.csv").exists()
