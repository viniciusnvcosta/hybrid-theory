"""DVC entry-point for ensemble stage.

This orchestrates the complete L1→L2→L3 pipeline and logs results.
"""

from pathlib import Path

from hydra import compose, initialize_config_dir

from cdade.ensemble.cdade import run_ensemble

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Initialize Hydra
with initialize_config_dir(config_dir=str(_PROJECT_ROOT / "configs"), version_base=None):
    cfg = compose(config_name="config")

# Run pipeline
results = run_ensemble(cfg)

# Print summary
print("\n=== CDADE Pipeline Complete ===")
print(f"Detectors: {results.get('detectors', {}).get('detector_count', 'N/A')}")
print(f"Reconciled: {results.get('reconciliation', {}).get('reconciled_count', 'N/A')}")
print(f"Selected: {results.get('selection', {}).get('selected_detectors', 'N/A')}")
competence = results.get("selection", {}).get("competence")
mean_comp = float(competence.mean()) if competence is not None and competence.size > 0 else "N/A"
print(f"Mean Competence: {mean_comp}")
print(f"MLflow Run ID: {results.get('mlflow_run_id', 'N/A')}")
print("=== End ===\n")
