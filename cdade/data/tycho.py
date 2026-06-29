"""Project Tycho v2.0 data loader — US Malaria (SNOMED 61462000).

Hierarchy: 56 US states/territories (leaves) → national (aggregate).

Raw file expected (relative to ``raw_dir``):
- ``US.61462000.csv`` — weekly state-level malaria case counts (1951–2017)

All outputs are in **counts** (incident cases per month per state).
Cumulative series rows are dropped before aggregation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

_NATIONAL = "US"
_CSV_FILENAME = "US.61462000.csv"

# Columns needed from the raw file
_USE_COLS = [
    "Admin1Name",
    "PeriodStartDate",
    "PartOfCumulativeCountSeries",
    "CountValue",
]


def load_raw(raw_dir: Path | str) -> pd.DataFrame:
    """Read the Tycho US Malaria CSV and return incident rows only.

    Drops rows where ``PartOfCumulativeCountSeries == 1`` (cumulative
    running totals). Sums across all subpopulations and age ranges.

    Args:
        raw_dir: Directory containing ``US.61462000.csv``.

    Returns:
        DataFrame with columns ``["Admin1Name", "PeriodStartDate", "CountValue"]``
        and a parsed ``PeriodStartDate`` column (datetime).
    """
    raw_dir = Path(raw_dir)
    df = pd.read_csv(
        raw_dir / _CSV_FILENAME,
        usecols=_USE_COLS + ["PeriodEndDate"],  # PeriodEndDate unused but harmless
        parse_dates=["PeriodStartDate"],
        low_memory=False,
    )
    df = df[df["PartOfCumulativeCountSeries"] == 0].copy()
    df = df.drop(columns=["PartOfCumulativeCountSeries", "PeriodEndDate"])
    return df


def build_hierarchy_spec(df: pd.DataFrame) -> dict:
    """Return the hierarchy specification derived from the loaded data.

    Args:
        df: DataFrame returned by :func:`load_raw`.

    Returns:
        Dict with keys ``national`` (str) and ``leaves`` (list[str], sorted).
    """
    leaves = sorted(df["Admin1Name"].unique().tolist())
    return {"national": _NATIONAL, "leaves": leaves}


def build_summing_matrix(spec: dict) -> np.ndarray:
    """Construct the summing matrix S for the Tycho hierarchy.

    Shape: ``(n_leaves + 1, n_leaves)``.
    Row 0 is the all-ones vector (national = sum of all leaves).
    Rows 1..n_leaves are identity rows.

    Args:
        spec: Hierarchy spec from :func:`build_hierarchy_spec`.

    Returns:
        Summing matrix of shape ``(n_leaves + 1, n_leaves)``.
    """
    n = len(spec["leaves"])
    top_row = np.ones((1, n), dtype=np.float64)
    leaf_rows = np.eye(n, dtype=np.float64)
    return np.vstack([top_row, leaf_rows])


def prepare_counts(df: pd.DataFrame, freq: str = "MS") -> pd.DataFrame:
    """Resample weekly incident counts to monthly and pivot to wide format.

    Args:
        df: DataFrame from :func:`load_raw`.
        freq: Pandas offset alias for target frequency (default ``"MS"``
              = month-start).

    Returns:
        DataFrame of shape ``(n_months, n_states)`` with integer counts.
        Index is a ``DatetimeIndex`` at the requested frequency.
        Columns are state names in sorted order (matches :func:`build_hierarchy_spec`).
    """
    df = df.copy()
    df = df.set_index("PeriodStartDate")
    # Group by state and resample to monthly sums
    monthly = df.groupby("Admin1Name")["CountValue"].resample(freq).sum().reset_index()
    pivot = monthly.pivot(index="PeriodStartDate", columns="Admin1Name", values="CountValue")
    pivot = pivot.sort_index().sort_index(axis=1)
    pivot = pivot.fillna(0).astype(np.int64)
    pivot.index = pd.DatetimeIndex(pivot.index, freq=freq)
    pivot.index.name = "Date"
    pivot.columns.name = None
    return pivot
