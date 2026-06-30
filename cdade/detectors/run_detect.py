"""CLI entry point for detector pipeline stage (DVC: detect).

Runs all registered detectors on injected data and saves results.
"""

from pathlib import Path

import pandas as pd
from omegaconf import DictConfig


def run_detect(cfg: DictConfig) -> dict:
    """Run all detectors on injected data.

    Args:
        cfg: Hydra configuration object

    Returns:
        Dictionary containing detector results
    """
    from cdade.registry import get_detector

    # Load injected data
    injected_path = Path("../data/injected/sivep_counts_injected.parquet")
    if not injected_path.exists():
        raise FileNotFoundError(f"Injected data not found: {injected_path}")

    data = pd.read_parquet(injected_path)

    # Get detector config
    detector_name = cfg.detector.name
    detector = get_detector(detector_name)

    # Fit and score
    detector.fit(data)
    scores = detector.score(data)

    # Save results
    output_dir = Path("../results/detectors")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save scores
    scores.to_csv(output_dir / "leaf_forecasts.csv", index=False)

    # Save detector results
    detector_results = {
        "name": detector_name,
        "scores": scores.values.tolist(),
    }

    import json

    with open(output_dir / "detector_results.json", "w") as f:
        json.dump(detector_results, f, indent=2)

    print(f"Detector results saved to {output_dir}")

    return {
        "detector_results": detector_results,
        "scores": scores,
        "detector_count": 1,
    }
