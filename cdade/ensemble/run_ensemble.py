"""DVC entry-point for ensemble stage.

This orchestrates the complete L1→L2→L3 pipeline and logs results.
"""

from hydra import compose, initialize_config_dir

from cdade.ensemble import run_ensemble

# Initialize Hydra
with initialize_config_dir(config_dir="configs", version_base=None):
    cfg = compose(config_name="config")

# Run pipeline
results = run_ensemble(cfg)

# Print summary
print("\n=== CDADE Pipeline Complete ===")
print(f"Detectors: {results.get('detectors', {}).get('detector_count', 'N/A')}")
print(f"Reconciled: {results.get('reconciliation', {}).get('reconciled_count', 'N/A')}")
print(f"Selected: {results.get('selection', {}).get('selected_detectors', 'N/A')}")
mean_comp = results.get("selection", {}).get("competence", {}).get("mean_competence", "N/A")
print(f"Mean Competence: {mean_comp}")
print(f"MLflow Run ID: {results.get('mlflow_run_id', 'N/A')}")
print("=== End ===\n")
