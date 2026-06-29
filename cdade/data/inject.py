"""Anomaly injection stage — reads processed Parquet files and writes injected copies.

CLI entry point: ``python -m cdade.data.inject``

For each HR/state column in the processed count matrices, applies
:func:`cdade.data.synthetic.inject_random` and writes the result alongside
a boolean mask DataFrame.

Outputs (written to ``data/injected/``):
- ``sivep_counts_injected.parquet`` — HR counts with synthetic anomalies
- ``sivep_counts_mask.parquet``     — boolean mask (True = anomalous)
- ``sivep_state_injected.parquet``  — state counts with synthetic anomalies
- ``sivep_state_mask.parquet``
- ``tycho_counts_injected.parquet`` — Tycho counts with synthetic anomalies
- ``tycho_counts_mask.parquet``
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from cdade.data.synthetic import inject_random

logger = logging.getLogger(__name__)

# Default injection parameters (overridden by configs/dataset/inject.yaml at
# the DVC stage level; these are the fallback when running the module directly)
_DEFAULTS = {
    "seed": 42,
    "contamination": 0.05,
    "spike_magnitude": 3.0,
    "level_shift_delta": 2.0,
    "drift_slope": 0.05,
}


def _inject_dataframe(
    df: pd.DataFrame,
    rng: np.random.Generator,
    contamination: float,
    spike_magnitude: float,
    level_shift_delta: float,
    drift_slope: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply :func:`inject_random` independently to each column.

    Args:
        df: Input DataFrame (rows = time, columns = series).
        rng: Random number generator (shared; state advances per column).
        contamination: Fraction of each series to perturb.
        spike_magnitude: Spike height in units of series std.
        level_shift_delta: Level-shift magnitude in units of series std.
        drift_slope: Per-step drift in units of series mean.

    Returns:
        Tuple ``(injected_df, mask_df)`` with same shape and index/columns as *df*.
    """
    injected_cols: dict[str, np.ndarray] = {}
    mask_cols: dict[str, np.ndarray] = {}
    for col in df.columns:
        series = df[col].to_numpy(dtype=float)
        inj, mask = inject_random(
            series,
            rng=rng,
            contamination=contamination,
            spike_magnitude=spike_magnitude,
            level_shift_delta=level_shift_delta,
            drift_slope=drift_slope,
        )
        injected_cols[col] = inj
        mask_cols[col] = mask

    return (
        pd.DataFrame(injected_cols, index=df.index),
        pd.DataFrame(mask_cols, index=df.index),
    )


def run(
    processed_dir: Path,
    out_dir: Path,
    seed: int = _DEFAULTS["seed"],
    contamination: float = _DEFAULTS["contamination"],
    spike_magnitude: float = _DEFAULTS["spike_magnitude"],
    level_shift_delta: float = _DEFAULTS["level_shift_delta"],
    drift_slope: float = _DEFAULTS["drift_slope"],
) -> None:
    """Inject synthetic anomalies into all processed count files.

    Args:
        processed_dir: ``data/processed/`` directory produced by the prepare stage.
        out_dir: Destination directory (``data/injected/``); created if absent.
        seed: Random seed for full reproducibility.
        contamination: Fraction of each series to perturb.
        spike_magnitude: Spike height in units of series std.
        level_shift_delta: Level-shift magnitude in units of series std.
        drift_slope: Per-step drift in units of series mean.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    inject_kwargs = {
        "contamination": contamination,
        "spike_magnitude": spike_magnitude,
        "level_shift_delta": level_shift_delta,
        "drift_slope": drift_slope,
    }

    # ── SIVEP HR counts ────────────────────────────────────────────────────
    sivep_counts = pd.read_parquet(processed_dir / "sivep_counts.parquet")
    inj, mask = _inject_dataframe(sivep_counts, rng, **inject_kwargs)
    inj.to_parquet(out_dir / "sivep_counts_injected.parquet")
    mask.to_parquet(out_dir / "sivep_counts_mask.parquet")
    logger.info("SIVEP HR: injected %d anomalous cells", mask.values.sum())

    # ── SIVEP state counts ─────────────────────────────────────────────────
    sivep_state = pd.read_parquet(processed_dir / "sivep_state.parquet")
    inj_s, mask_s = _inject_dataframe(sivep_state, rng, **inject_kwargs)
    inj_s.to_parquet(out_dir / "sivep_state_injected.parquet")
    mask_s.to_parquet(out_dir / "sivep_state_mask.parquet")
    logger.info("SIVEP state: injected %d anomalous cells", mask_s.values.sum())

    # ── Tycho counts ───────────────────────────────────────────────────────
    tycho_counts = pd.read_parquet(processed_dir / "tycho_counts.parquet")
    inj_t, mask_t = _inject_dataframe(tycho_counts, rng, **inject_kwargs)
    inj_t.to_parquet(out_dir / "tycho_counts_injected.parquet")
    mask_t.to_parquet(out_dir / "tycho_counts_mask.parquet")
    logger.info("Tycho: injected %d anomalous cells", mask_t.values.sum())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run(
        processed_dir=Path("data/processed"),
        out_dir=Path("data/injected"),
    )
