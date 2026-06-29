"""Tests for cdade/data/sivep.py."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cdade.data.sivep import (
    build_hierarchy_spec,
    build_summing_matrix,
    load_raw,
    prepare_counts,
    prepare_state_counts,
)

RAW_DIR = Path("data/raw")


@pytest.fixture(scope="module")
def raw_data():
    state_df, hr_df = load_raw(RAW_DIR)
    return state_df, hr_df


@pytest.fixture(scope="module")
def counts(raw_data):
    _, hr_df = raw_data
    return prepare_counts(hr_df)


@pytest.fixture(scope="module")
def state_counts(raw_data):
    state_df, _ = raw_data
    return prepare_state_counts(state_df)


# ── load_raw ──────────────────────────────────────────────────────────────────


def test_load_raw_returns_two_dataframes(raw_data):
    state_df, hr_df = raw_data
    assert isinstance(state_df, pd.DataFrame)
    assert isinstance(hr_df, pd.DataFrame)


def test_state_df_has_positives_column(raw_data):
    state_df, _ = raw_data
    assert "positives" in state_df.columns


def test_hr_df_has_required_columns(raw_data):
    _, hr_df = raw_data
    for col in (
        "notification.year",
        "notification.month",
        "notification.hr",
        "exam.result",
        "testperhr",
    ):
        assert col in hr_df.columns


# ── build_hierarchy_spec ──────────────────────────────────────────────────────


def test_hierarchy_spec_keys():
    spec = build_hierarchy_spec()
    assert set(spec.keys()) == {"state", "leaves"}


def test_hierarchy_spec_state():
    spec = build_hierarchy_spec()
    assert spec["state"] == "PA"


def test_hierarchy_spec_leaves_count():
    spec = build_hierarchy_spec()
    assert len(spec["leaves"]) == 13


# ── build_summing_matrix ──────────────────────────────────────────────────────


def test_summing_matrix_shape():
    S = build_summing_matrix(13)
    assert S.shape == (14, 13)


def test_summing_matrix_top_row_is_ones():
    S = build_summing_matrix(13)
    np.testing.assert_array_equal(S[0], np.ones(13))


def test_summing_matrix_leaf_rows_form_identity():
    S = build_summing_matrix(13)
    np.testing.assert_array_equal(S[1:], np.eye(13))


# ── prepare_counts ────────────────────────────────────────────────────────────


def test_counts_shape(counts):
    assert counts.shape == (132, 13), f"Expected (132, 13), got {counts.shape}"


def test_counts_columns_are_canonical_leaves(counts):
    spec = build_hierarchy_spec()
    assert list(counts.columns) == spec["leaves"]


def test_counts_index_is_monthly(counts):
    assert counts.index.freq == "MS"


def test_counts_no_nan(counts):
    assert not counts.isna().any().any()


def test_counts_non_negative(counts):
    assert (counts >= 0).all().all()


# ── prepare_state_counts ──────────────────────────────────────────────────────


def test_state_counts_length(state_counts):
    assert len(state_counts) == 132


def test_state_counts_name(state_counts):
    assert state_counts.name == "PA"


def test_state_counts_no_nan(state_counts):
    assert not state_counts.isna().any()


# ── coherence: leaves ≈ state aggregate ───────────────────────────────────────


def test_leaf_sum_approximates_state(counts, state_counts):
    """Sum of 13 HR positive counts must be close to the PA state positive count.

    The two series are derived from different raw files so minor discrepancies
    (<= 5%) are acceptable (reporting lag, rounding, missing records).
    """
    leaf_sum = counts.sum(axis=1)
    # Align on common index
    common = leaf_sum.index.intersection(state_counts.index)
    diff = (leaf_sum[common] - state_counts[common]).abs()
    rel_err = diff / state_counts[common].clip(lower=1)
    assert rel_err.median() < 0.05, f"Median relative error {rel_err.median():.3f} exceeds 5%"
