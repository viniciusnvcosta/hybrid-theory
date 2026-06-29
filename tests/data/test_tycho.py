"""Tests for cdade/data/tycho.py."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cdade.data.tycho import (
    build_hierarchy_spec,
    build_summing_matrix,
    load_raw,
    prepare_counts,
)

RAW_DIR = Path("data/raw/US.61462000")


@pytest.fixture(scope="module")
def raw_df():
    return load_raw(RAW_DIR)


@pytest.fixture(scope="module")
def counts(raw_df):
    return prepare_counts(raw_df)


@pytest.fixture(scope="module")
def spec(raw_df):
    return build_hierarchy_spec(raw_df)


# ── load_raw ──────────────────────────────────────────────────────────────────


def test_load_raw_returns_dataframe(raw_df):
    assert isinstance(raw_df, pd.DataFrame)


def test_load_raw_no_cumulative_rows(raw_df):
    # PartOfCumulativeCountSeries column is dropped after filtering
    assert "PartOfCumulativeCountSeries" not in raw_df.columns


def test_load_raw_required_columns(raw_df):
    for col in ("Admin1Name", "PeriodStartDate", "CountValue"):
        assert col in raw_df.columns


def test_load_raw_count_value_non_negative(raw_df):
    assert (raw_df["CountValue"] >= 0).all()


# ── build_hierarchy_spec ──────────────────────────────────────────────────────


def test_hierarchy_spec_keys(spec):
    assert set(spec.keys()) == {"national", "leaves"}


def test_hierarchy_spec_national(spec):
    assert spec["national"] == "US"


def test_hierarchy_spec_leaves_non_empty(spec):
    assert len(spec["leaves"]) > 0


def test_hierarchy_spec_leaves_sorted(spec):
    assert spec["leaves"] == sorted(spec["leaves"])


# ── build_summing_matrix ──────────────────────────────────────────────────────


def test_summing_matrix_shape(spec):
    n = len(spec["leaves"])
    S = build_summing_matrix(spec)
    assert S.shape == (n + 1, n)


def test_summing_matrix_top_row_is_ones(spec):
    n = len(spec["leaves"])
    S = build_summing_matrix(spec)
    np.testing.assert_array_equal(S[0], np.ones(n))


def test_summing_matrix_leaf_rows_form_identity(spec):
    n = len(spec["leaves"])
    S = build_summing_matrix(spec)
    np.testing.assert_array_equal(S[1:], np.eye(n))


# ── prepare_counts ────────────────────────────────────────────────────────────


def test_counts_returns_dataframe(counts):
    assert isinstance(counts, pd.DataFrame)


def test_counts_index_is_monthly(counts):
    assert counts.index.freq == "MS"


def test_counts_columns_match_spec(counts, spec):
    assert list(counts.columns) == spec["leaves"]


def test_counts_no_nan(counts):
    assert not counts.isna().any().any()


def test_counts_non_negative(counts):
    assert (counts >= 0).all().all()


def test_counts_date_index(counts):
    assert isinstance(counts.index, pd.DatetimeIndex)


# ── coherence: S @ leaf_counts.T has row 0 == national sum ───────────────────


def test_summing_matrix_coherence(counts, spec):
    """Applying S to leaf counts should reproduce the national aggregate in row 0."""
    S = build_summing_matrix(spec)
    result = S @ counts.values.T  # (n+1, n_months)
    national_via_S = result[0]
    national_direct = counts.sum(axis=1).values
    np.testing.assert_array_equal(national_via_S, national_direct)
