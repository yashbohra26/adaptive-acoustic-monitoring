"""Candidate lightweight features investigated for the Phase 4 adaptive selector.

This module provides candidate lightweight features explored to address
the trade-off between frequency sensitivity and computational work.
All operation counts are explicitly documented.
"""
from __future__ import annotations

import numpy as np
from features import zero_crossing_frequency


def normalized_difference_ratio(x: np.ndarray) -> float:
    """Compute normalized lag-1 difference energy ratio.

    R_diff = sum((x[n] - x[n-1])^2) / sum((x[n] - mean(x))^2)

    For a pure sinusoid x[n] = A * sin(omega * n):
        E[R_diff] = 2 * (1 - cos(2 * pi * f / sample_rate))
    Monotonically increases with frequency for f < sample_rate / 2.

    Operation count proxy:
    - Mean and centering: 2 * N ops
    - Variance denominator: 2 * N ops
    - First difference: N ops
    - Squared difference numerator: 2 * N ops
    - Division: 1 op
    Total: 7 * N + 1 ops (~7,169 ops for N=1024).

    Limitation:
    At low SNR (e.g. 0 dB), white noise has R_diff -> 2.0 (uncorrelated samples),
    which smears normal and frequency-shifted signals into the same range (~1.0 to 1.25),
    making threshold separation unviable under severe noise without escalation.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 2:
        return 0.0

    centered = x - np.mean(x)
    var = float(np.sum(centered**2))
    if var <= 1e-12:
        return 0.0

    diff = np.diff(x)
    diff_energy = float(np.sum(diff**2))
    return diff_energy / var


def get_clean_calibrated_zcf_bounds(
    calib_signals: list[np.ndarray],
    calib_snrs: list[float],
    sample_rate_hz: int = 4000,
    snr_clean_threshold_db: float = 20.0,
    margin_hz: float = 10.0,
) -> tuple[float, float]:
    """Derive tight normal frequency bounds from clean threshold-calibration windows.

    By filtering calibration windows to clean SNRs (>= 20 dB), we capture the true
    machine fundamental variation without noise-induced sign changes that inflate ZCF.

    Operation count at inference time:
    Zero additional operations, because zero_crossing_frequency is already
    computed during Stage 1 Fixed-Low processing.
    """
    clean_zcfs = [
        zero_crossing_frequency(sig, sample_rate_hz)
        for sig, snr in zip(calib_signals, calib_snrs)
        if snr >= snr_clean_threshold_db
    ]
    if not clean_zcfs:
        raise ValueError("No clean calibration signals found meeting the SNR threshold")

    z_min = float(min(clean_zcfs)) - margin_hz
    z_max = float(max(clean_zcfs)) + margin_hz
    return z_min, z_max
