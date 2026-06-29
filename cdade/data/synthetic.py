"""Synthetic anomaly injection for time series.

Provides three primitive anomaly types (spike, level shift, drift)
and a combined random injector that mixes all three.

All functions are **pure and stateless** — randomness is supplied via
a caller-provided ``numpy.random.Generator`` so that seeding is handled
externally (typically from ``configs/dataset/inject.yaml``).

Each primitive returns ``(injected_series, bool_mask)`` where the mask
marks positions that were altered by injection.
"""

from __future__ import annotations

import numpy as np


def inject_spike(
    series: np.ndarray,
    idx: int,
    magnitude: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Inject a single spike anomaly at position *idx*.

    The spike amplitude is ``magnitude * local_std`` where ``local_std``
    is the standard deviation of the full series.  Direction (up/down)
    is chosen randomly.

    Args:
        series: 1-D float array of the original time series.
        idx: Position to inject the spike (0-indexed).
        magnitude: Spike height in units of series std.
        rng: Seeded random number generator.

    Returns:
        Tuple ``(injected, mask)`` where *mask* is True at *idx*.
    """
    out = series.copy().astype(float)
    std = float(np.std(series)) or 1.0
    direction = rng.choice([-1.0, 1.0])
    out[idx] += direction * magnitude * std
    mask = np.zeros(len(series), dtype=bool)
    mask[idx] = True
    return out, mask


def inject_level_shift(
    series: np.ndarray,
    start_idx: int,
    delta: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Inject a persistent level shift starting at *start_idx*.

    All points from *start_idx* to the end are shifted by
    ``delta * series_std``.  Direction is chosen randomly.

    Args:
        series: 1-D float array of the original time series.
        start_idx: First position affected by the shift.
        delta: Shift magnitude in units of series std.
        rng: Seeded random number generator.

    Returns:
        Tuple ``(injected, mask)`` where *mask* is True for all shifted positions.
    """
    out = series.copy().astype(float)
    std = float(np.std(series)) or 1.0
    direction = rng.choice([-1.0, 1.0])
    out[start_idx:] += direction * delta * std
    mask = np.zeros(len(series), dtype=bool)
    mask[start_idx:] = True
    return out, mask


def inject_drift(
    series: np.ndarray,
    start_idx: int,
    slope: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Inject a linear drift starting at *start_idx*.

    Each step after *start_idx* accumulates an additional ``slope * mean``
    offset, creating a gradual upward or downward trend.

    Args:
        series: 1-D float array of the original time series.
        start_idx: First position affected by the drift.
        slope: Per-step drift in units of series mean.
        rng: Seeded random number generator.

    Returns:
        Tuple ``(injected, mask)`` where *mask* is True for all drifted positions.
    """
    out = series.copy().astype(float)
    mean = float(np.mean(series)) or 1.0
    direction = rng.choice([-1.0, 1.0])
    n_affected = len(series) - start_idx
    ramp = np.arange(1, n_affected + 1, dtype=float) * slope * mean * direction
    out[start_idx:] += ramp
    mask = np.zeros(len(series), dtype=bool)
    mask[start_idx:] = True
    return out, mask


def inject_random(
    series: np.ndarray,
    rng: np.random.Generator,
    contamination: float = 0.05,
    spike_magnitude: float = 3.0,
    level_shift_delta: float = 2.0,
    drift_slope: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Inject a mix of random anomalies into *series*.

    Randomly selects ``round(contamination * len(series))`` positions and
    applies one of the three anomaly types to each.  Anomaly type is drawn
    uniformly: spike (1/3), level shift (1/3), drift (1/3).

    Args:
        series: 1-D float array of the original time series.
        rng: Seeded random number generator (controls reproducibility).
        contamination: Fraction of series length to perturb (default 0.05).
        spike_magnitude: Spike height in units of series std.
        level_shift_delta: Level-shift magnitude in units of series std.
        drift_slope: Per-step drift in units of series mean.

    Returns:
        Tuple ``(injected, mask)`` where *mask* marks all anomalous positions.
    """
    n = len(series)
    n_anomalies = max(1, round(contamination * n))

    indices = rng.choice(n, size=n_anomalies, replace=False)
    anomaly_types = rng.integers(0, 3, size=n_anomalies)  # 0=spike,1=shift,2=drift

    out = series.copy().astype(float)
    mask = np.zeros(n, dtype=bool)

    for idx, atype in zip(indices, anomaly_types, strict=True):
        if atype == 0:
            segment, seg_mask = inject_spike(out, int(idx), spike_magnitude, rng)
        elif atype == 1:
            segment, seg_mask = inject_level_shift(out, int(idx), level_shift_delta, rng)
        else:
            segment, seg_mask = inject_drift(out, int(idx), drift_slope, rng)
        out = segment
        mask |= seg_mask

    return out, mask
