# ABOUTME: CLI entry point for the detector pipeline stage (DVC: detect).
# ABOUTME: Writes detector scores and results to results/detectors/{dataset}/.

"""CLI entry point for detector pipeline stage (DVC: detect).

Runs all registered detectors on injected data and saves results.
"""

import json
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig

from cdade.data.dataset_paths import get_dataset_artifact_paths

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def run_detect(cfg: DictConfig, dataset_name: str | None = None) -> dict:
    """Run all detectors on injected data for one dataset.

    Args:
        cfg: Hydra configuration object.
        dataset_name: Dataset to process. If None, inferred from cfg.

    Returns:
        Dictionary containing detector results.
    """
    import cdade.detectors.cblof  # noqa: F401
    import cdade.detectors.cof  # noqa: F401
    import cdade.detectors.hbos  # noqa: F401
    import cdade.detectors.iforest  # noqa: F401
    import cdade.detectors.knn  # noqa: F401
    import cdade.detectors.lof  # noqa: F401
    import cdade.detectors.mcd  # noqa: F401
    import cdade.detectors.ocsvm  # noqa: F401
    import cdade.detectors.pca  # noqa: F401
    import cdade.detectors.sos  # noqa: F401
    from cdade.registry import get_detector

    if dataset_name is None:
        from cdade.data.dataset_paths import _dataset_name as _dn

        dataset_name = _dn(cfg)

    # Load injected data
    artifact_paths = get_dataset_artifact_paths(dataset_name, project_root=_PROJECT_ROOT)
    injected_path = artifact_paths["injected_counts"]
    if not injected_path.exists():
        raise FileNotFoundError(f"Injected data not found: {injected_path}")

    data = pd.read_parquet(injected_path)

    # Get detector config
    detector_name = cfg.detector.name
    detector_cls = get_detector(detector_name)

    # Instantiate using the detector-specific config object if available.
    detector_cfg = getattr(cfg.detector, "config", None)
    if detector_cfg is None:
        try:
            detector = detector_cls(cfg=cfg.detector)
        except TypeError:
            detector = detector_cls()
    else:
        detector = detector_cls(detector_cfg)

    # Fit and score
    detector.fit(data)
    scores = detector.score(data)
    scores_df = pd.DataFrame(scores, columns=["score"])

    # Save results to namespaced directory
    output_dir = _PROJECT_ROOT / "results" / "detectors" / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save scores
    scores_df.to_csv(output_dir / "leaf_forecasts.csv", index=False)

    # Save detector results
    detector_results = {
        "name": detector_name,
        "scores": scores_df["score"].tolist(),
    }

    with open(output_dir / "detector_results.json", "w") as f:
        json.dump(detector_results, f, indent=2)

    return {
        "detector_results": detector_results,
        "scores": scores,
        "detector_count": 1,
    }


@hydra.main(
    config_path=str(_PROJECT_ROOT / "configs"),
    config_name="config",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    """CLI entry point for the detect stage — loops over datasets.active."""
    from cdade.data.dataset_paths import _iter_datasets

    for dataset_name, _paths in _iter_datasets(cfg, project_root=_PROJECT_ROOT):
        run_detect(cfg, dataset_name=dataset_name)


if __name__ == "__main__":
    main()
