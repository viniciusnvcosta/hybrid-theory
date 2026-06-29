"""SIVEP-Malária data loader for Pará (PA) state, Brazil.

Hierarchy: 13 health regions (leaves) → PA state (aggregate).

All outputs are in **counts** (integer positive cases per month per region).
Proportions must be derived post-hoc; never reconcile on proportions.

Raw files expected (relative to ``raw_dir``):
- ``PA.csv``                — state-level monthly aggregates (2009–2019)
- ``PASIVEPDailyPerHr.csv`` — HR-level monthly records by exam result (2009–2019)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Canonical leaf order — matches configs/dataset/sivep.yaml
_LEAVES: list[str] = [
    "ARAGUAIA",
    "BAIXO AMAZONAS",
    "CARAJAS",
    "LAGO DE TUCURUI",
    "MARAJO I",
    "MARAJO II",
    "METROPOLITANA I",
    "METROPOLITANA II",
    "METROPOLITANA III",
    "RIO CAETES",
    "TAPAJOS",
    "TOCANTINS",
    "XINGU",
]

_STATE = "PA"
_NEGATIVE_LABEL = "negative"


def load_raw(raw_dir: Path | str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read both raw CSV files for the Pará dataset.

    Args:
        raw_dir: Directory containing ``PA.csv`` and ``PASIVEPDailyPerHr.csv``.

    Returns:
        Tuple ``(state_df, hr_df)`` where:
        - ``state_df`` is the monthly state-level DataFrame (index = Date).
        - ``hr_df`` is the raw HR-level records DataFrame.
    """
    raw_dir = Path(raw_dir)
    state_df = pd.read_csv(
        raw_dir / "PA.csv",
        index_col=0,
        parse_dates=True,
    )
    hr_df = pd.read_csv(raw_dir / "PASIVEPDailyPerHr.csv")
    return state_df, hr_df


def build_hierarchy_spec() -> dict:
    """Return the fixed hierarchy specification for the SIVEP-PA dataset.

    Returns:
        Dict with keys ``state`` (str) and ``leaves`` (list[str]).
    """
    return {"state": _STATE, "leaves": list(_LEAVES)}


def build_summing_matrix(n_leaves: int = 13) -> np.ndarray:
    """Construct the summing matrix S for the SIVEP hierarchy.

    The matrix has shape ``(n_leaves + 1, n_leaves)``.
    Row 0 is the all-ones vector (state = sum of all leaves).
    Rows 1..n_leaves are identity rows (each leaf maps to itself).

    Args:
        n_leaves: Number of leaf series (default 13 for Pará).

    Returns:
        Summing matrix of shape ``(n_leaves + 1, n_leaves)``.
    """
    top_row = np.ones((1, n_leaves), dtype=np.float64)
    leaf_rows = np.eye(n_leaves, dtype=np.float64)
    return np.vstack([top_row, leaf_rows])


def prepare_counts(hr_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot HR-level records into a monthly positive-count matrix.

    Positives are all rows where ``exam.result != 'negative'``.
    The result is a DataFrame indexed by month-start dates with one
    column per health region in canonical leaf order.

    Args:
        hr_df: Raw HR DataFrame from :func:`load_raw`.

    Returns:
        DataFrame of shape ``(n_months, 13)`` with integer counts.
        Index is monthly ``DatetimeIndex`` (freq=MS).
        Columns follow :data:`_LEAVES` order.
    """
    pos = hr_df[hr_df["exam.result"] != _NEGATIVE_LABEL].copy()
    pos["Date"] = pd.to_datetime(
        pos[["notification.year", "notification.month"]]
        .rename(columns={"notification.year": "year", "notification.month": "month"})
        .assign(day=1)
    )
    monthly = pos.groupby(["Date", "notification.hr"])["testperhr"].sum().reset_index()
    pivot = monthly.pivot(index="Date", columns="notification.hr", values="testperhr")
    pivot = pivot.reindex(columns=_LEAVES).fillna(0).astype(np.int64)
    pivot.index = pd.DatetimeIndex(pivot.index, freq="MS")
    pivot.index.name = "Date"
    pivot.columns.name = None
    return pivot


def prepare_state_counts(state_df: pd.DataFrame) -> pd.Series:
    """Extract the monthly positive count series for PA state.

    Args:
        state_df: State DataFrame from :func:`load_raw`.

    Returns:
        Series of positive counts indexed by month-start dates (freq=MS),
        named ``"PA"``.
    """
    series = state_df["positives"].copy()
    series.index = pd.DatetimeIndex(series.index, freq="MS")
    series.name = "PA"
    return series
