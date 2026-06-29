"""Data preparation stage — reads raw CSVs and writes processed Parquet files.

CLI entry point: ``python -m cdade.data.prepare``

Outputs (written to ``data/processed/``):
- ``sivep_counts.parquet``   — (n_months, 13) HR positive counts
- ``sivep_state.parquet``    — (n_months,) PA state positive counts
- ``tycho_counts.parquet``   — (n_months, n_states) monthly incident counts
- ``hierarchy_sivep.json``   — hierarchy spec for SIVEP-PA
- ``hierarchy_tycho.json``   — hierarchy spec for Tycho US Malaria
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from cdade.data.sivep import (
    build_hierarchy_spec as sivep_hierarchy,
)
from cdade.data.sivep import (
    load_raw as sivep_load_raw,
)
from cdade.data.sivep import (
    prepare_counts as sivep_prepare_counts,
)
from cdade.data.sivep import (
    prepare_state_counts as sivep_prepare_state_counts,
)
from cdade.data.tycho import (
    build_hierarchy_spec as tycho_hierarchy,
)
from cdade.data.tycho import (
    load_raw as tycho_load_raw,
)
from cdade.data.tycho import (
    prepare_counts as tycho_prepare_counts,
)

logger = logging.getLogger(__name__)


def run(raw_dir: Path, out_dir: Path) -> None:
    """Prepare both datasets and write outputs to *out_dir*.

    Args:
        raw_dir: Root of ``data/raw/`` directory.
        out_dir: Destination directory (``data/processed/``); created if absent.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── SIVEP-Malária ──────────────────────────────────────────────────────
    logger.info("Loading SIVEP-Malária raw data …")
    state_df, hr_df = sivep_load_raw(raw_dir)

    counts = sivep_prepare_counts(hr_df)
    state_counts = sivep_prepare_state_counts(state_df)
    spec_sivep = sivep_hierarchy()

    counts.to_parquet(out_dir / "sivep_counts.parquet")
    state_counts.to_frame().to_parquet(out_dir / "sivep_state.parquet")
    (out_dir / "hierarchy_sivep.json").write_text(
        json.dumps(spec_sivep, ensure_ascii=False, indent=2)
    )
    logger.info(
        "SIVEP: wrote %s × %s counts, %s state rows",
        counts.shape[0],
        counts.shape[1],
        len(state_counts),
    )

    # ── Project Tycho ──────────────────────────────────────────────────────
    logger.info("Loading Project Tycho raw data …")
    tycho_raw = tycho_load_raw(raw_dir / "US.61462000")

    tycho_counts = tycho_prepare_counts(tycho_raw)
    spec_tycho = tycho_hierarchy(tycho_raw)

    tycho_counts.to_parquet(out_dir / "tycho_counts.parquet")
    (out_dir / "hierarchy_tycho.json").write_text(
        json.dumps(spec_tycho, ensure_ascii=False, indent=2)
    )
    logger.info(
        "Tycho: wrote %s × %s counts",
        tycho_counts.shape[0],
        tycho_counts.shape[1],
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run(
        raw_dir=Path("data/raw"),
        out_dir=Path("data/processed"),
    )
