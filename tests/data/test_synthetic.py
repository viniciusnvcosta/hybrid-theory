"""Tests for cdade/data/synthetic.py."""

import numpy as np
import pytest

from cdade.data.synthetic import (
    inject_drift,
    inject_level_shift,
    inject_random,
    inject_spike,
)

RNG_SEED = 42


@pytest.fixture
def rng():
    return np.random.default_rng(RNG_SEED)


@pytest.fixture
def series():
    rng = np.random.default_rng(0)
    return rng.integers(10, 100, size=60).astype(float)


# ── inject_spike ──────────────────────────────────────────────────────────────


def test_spike_output_shape(series, rng):
    out, mask = inject_spike(series, idx=10, magnitude=3.0, rng=rng)
    assert out.shape == series.shape
    assert mask.shape == series.shape


def test_spike_only_alters_target_index(series, rng):
    out, mask = inject_spike(series, idx=10, magnitude=3.0, rng=rng)
    assert mask.sum() == 1
    assert mask[10]
    np.testing.assert_array_equal(out[~mask], series[~mask])


def test_spike_changes_target_value(series, rng):
    out, mask = inject_spike(series, idx=10, magnitude=3.0, rng=rng)
    assert out[10] != series[10]


# ── inject_level_shift ────────────────────────────────────────────────────────


def test_level_shift_output_shape(series, rng):
    out, mask = inject_level_shift(series, start_idx=20, delta=2.0, rng=rng)
    assert out.shape == series.shape
    assert mask.shape == series.shape


def test_level_shift_affects_from_start(series, rng):
    start = 20
    out, mask = inject_level_shift(series, start_idx=start, delta=2.0, rng=rng)
    assert mask[:start].sum() == 0
    assert mask[start:].all()


def test_level_shift_unaffected_prefix(series, rng):
    start = 20
    out, mask = inject_level_shift(series, start_idx=start, delta=2.0, rng=rng)
    np.testing.assert_array_equal(out[:start], series[:start])


# ── inject_drift ──────────────────────────────────────────────────────────────


def test_drift_output_shape(series, rng):
    out, mask = inject_drift(series, start_idx=30, slope=0.05, rng=rng)
    assert out.shape == series.shape
    assert mask.shape == series.shape


def test_drift_affects_from_start(series, rng):
    start = 30
    out, mask = inject_drift(series, start_idx=start, slope=0.05, rng=rng)
    assert mask[:start].sum() == 0
    assert mask[start:].all()


def test_drift_unaffected_prefix(series, rng):
    start = 30
    out, mask = inject_drift(series, start_idx=start, slope=0.05, rng=rng)
    np.testing.assert_array_equal(out[:start], series[:start])


# ── inject_random ─────────────────────────────────────────────────────────────


def test_random_output_shape(series, rng):
    out, mask = inject_random(series, rng=rng, contamination=0.05)
    assert out.shape == series.shape
    assert mask.shape == series.shape


def test_random_mask_cardinality(series):
    contamination = 0.10
    rng = np.random.default_rng(RNG_SEED)
    _, mask = inject_random(series, rng=rng, contamination=contamination)
    # Mask may cover more positions than n_anomalies (level shift / drift expand)
    # but must cover at least the intended injection count
    expected_min = round(contamination * len(series))
    assert mask.sum() >= expected_min


def test_random_seed_stability(series):
    """Same seed → identical output."""
    out1, mask1 = inject_random(series, rng=np.random.default_rng(RNG_SEED))
    out2, mask2 = inject_random(series, rng=np.random.default_rng(RNG_SEED))
    np.testing.assert_array_equal(out1, out2)
    np.testing.assert_array_equal(mask1, mask2)


def test_random_different_seeds_differ(series):
    """Different seeds should almost certainly produce different outputs."""
    out1, _ = inject_random(series, rng=np.random.default_rng(1))
    out2, _ = inject_random(series, rng=np.random.default_rng(99))
    assert not np.array_equal(out1, out2)


def test_random_masked_positions_altered(series):
    """At least some masked positions must differ from the original."""
    out, mask = inject_random(series, rng=np.random.default_rng(RNG_SEED))
    if mask.sum() > 0:
        assert not np.array_equal(out[mask], series[mask])
