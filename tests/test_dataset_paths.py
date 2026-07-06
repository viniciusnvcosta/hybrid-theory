# ABOUTME: Tests for dataset_paths helpers including _iter_datasets generator.
# ABOUTME: Verifies routing logic for sivep/tycho and fallback behaviour.

from types import SimpleNamespace

from cdade.data.dataset_paths import _iter_datasets


class TestIterDatasets:
    def test_yields_both_active_datasets(self, tmp_path):
        cfg = SimpleNamespace(datasets=SimpleNamespace(active=["sivep", "tycho"]))
        results = list(_iter_datasets(cfg, project_root=tmp_path))
        names = [r[0] for r in results]
        assert names == ["sivep", "tycho"]

    def test_paths_are_namespaced_by_dataset(self, tmp_path):
        cfg = SimpleNamespace(datasets=SimpleNamespace(active=["sivep", "tycho"]))
        results = dict(_iter_datasets(cfg, project_root=tmp_path))
        assert "sivep_counts_injected" in str(results["sivep"]["injected_counts"])
        assert "tycho_counts_injected" in str(results["tycho"]["injected_counts"])

    def test_defaults_to_sivep_when_datasets_attr_absent(self, tmp_path):
        cfg = SimpleNamespace()  # no datasets attribute
        results = list(_iter_datasets(cfg, project_root=tmp_path))
        assert len(results) == 1
        assert results[0][0] == "sivep"

    def test_single_dataset_list(self, tmp_path):
        cfg = SimpleNamespace(datasets=SimpleNamespace(active=["tycho"]))
        results = list(_iter_datasets(cfg, project_root=tmp_path))
        assert len(results) == 1
        assert results[0][0] == "tycho"
