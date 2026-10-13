"""Phase 1 verification tests.

Covers:
1. Reproducibility under fixed seed.
2. Group split integrity (disjoint group IDs between dev and test; noise variants stay grouped).
3. Label and metadata schema correctness (anomaly labels, frequency target semantics).
4. Feature calculation behavior on known baseline signals (pure tones, DC, impulses).
"""
import sys
from pathlib import Path
import numpy as np
import pytest

# Ensure python directory is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

from config import DEFAULT_CONFIG, StudyConfig, SIGNAL_CASES
from features import (
    rms,
    crest_factor,
    kurtosis,
    zero_crossing_frequency,
    fft_peak_frequency,
    extract_features,
)
from generate_signals import build_dataset, synthesize_case


def test_reproducibility():
    """Verify that identical config seeds produce bitwise-identical records and signals."""
    records1, signals1 = build_dataset(groups=8, config=DEFAULT_CONFIG)
    records2, signals2 = build_dataset(groups=8, config=DEFAULT_CONFIG)

    assert len(records1) == len(records2)
    assert len(signals1) == len(signals2)

    for r1, r2 in zip(records1, records2):
        assert r1["group_id"] == r2["group_id"]
        assert r1["split"] == r2["split"]
        assert r1["case"] == r2["case"]
        assert r1["anomaly"] == r2["anomaly"]
        assert r1["snr_db"] == r2["snr_db"]
        assert r1["target_frequency_hz"] == r2["target_frequency_hz"]
        key = r1["signal_key"]
        np.testing.assert_array_equal(signals1[key], signals2[key])


def test_clean_recipe_preservation_across_snrs():
    """Verify that different SNRs share the same clean underlying signal for a given (group, case)."""
    # Test synthesizing across SNRs using the generation recipe logic
    group_id = "g0001"
    case = "normal"
    f0 = 220.0
    recipe_seed = DEFAULT_CONFIG.seed + sum(map(ord, group_id + case))

    # Two different SNRs
    noise_seed_30 = recipe_seed + int((30.0 + 50) * 100)
    noise_seed_10 = recipe_seed + int((10.0 + 50) * 100)

    sig_30, target_30 = synthesize_case(
        case, f0, 30.0, np.random.default_rng(recipe_seed), np.random.default_rng(noise_seed_30), DEFAULT_CONFIG
    )
    sig_10, target_10 = synthesize_case(
        case, f0, 10.0, np.random.default_rng(recipe_seed), np.random.default_rng(noise_seed_10), DEFAULT_CONFIG
    )

    assert target_30 == target_10 == f0
    # The noise realizations should differ, but signals should correlate strongly
    assert not np.array_equal(sig_30, sig_10)
    corr = np.corrcoef(sig_30, sig_10)[0, 1]
    assert corr > 0.90


def test_grouped_split_integrity():
    """Verify that source groups are strictly disjoint between development and test splits."""
    records, _ = build_dataset(groups=8, config=DEFAULT_CONFIG)

    dev_groups = {r["group_id"] for r in records if r["split"] == "development"}
    test_groups = {r["group_id"] for r in records if r["split"] == "test"}

    # Groups must be mutually disjoint
    assert dev_groups.isdisjoint(test_groups)

    # Every window from a given group must share the same split assignment
    group_splits = {}
    for r in records:
        gid = r["group_id"]
        split = r["split"]
        if gid in group_splits:
            assert group_splits[gid] == split, f"Group {gid} leaked across splits!"
        else:
            group_splits[gid] = split


def test_labels_and_target_frequency_specification():
    """Verify correct anomaly labels and target frequency rules across all cases."""
    records, _ = build_dataset(groups=8, config=DEFAULT_CONFIG)

    for r in records:
        case = r["case"]
        if case == "normal":
            assert r["anomaly"] == 0
            assert isinstance(r["target_frequency_hz"], (int, float))
            assert DEFAULT_CONFIG.normal_frequency_range_hz[0] <= r["target_frequency_hz"] <= DEFAULT_CONFIG.normal_frequency_range_hz[1]
        elif case == "frequency_shift":
            assert r["anomaly"] == 1
            assert isinstance(r["target_frequency_hz"], (int, float))
            min_shifted = DEFAULT_CONFIG.normal_frequency_range_hz[0] * DEFAULT_CONFIG.shift_ratio_range[0]
            max_shifted = DEFAULT_CONFIG.normal_frequency_range_hz[1] * DEFAULT_CONFIG.shift_ratio_range[1]
            assert min_shifted <= r["target_frequency_hz"] <= max_shifted
        elif case in ("added_harmonic", "impact"):
            assert r["anomaly"] == 1
            # Must have no single target frequency defined
            assert r["target_frequency_hz"] == ""
        else:
            pytest.fail(f"Unexpected case: {case}")


def test_feature_behavior_on_known_signals():
    """Verify feature behavior on well-defined mathematical reference signals."""
    sample_rate = 4000
    n = 1024
    t = np.arange(n) / sample_rate
    f0 = 250.0  # Exactly bin 64 for 4000 Hz / 1024 points (4000 * 64 / 1024 = 250.0)

    # 1. Pure sine wave: A * sin(2*pi*f0*t)
    amplitude = 1.0
    sine = amplitude * np.sin(2 * np.pi * f0 * t)

    sine_rms = rms(sine)
    # Theoretical RMS for sine of amplitude 1 is 1/sqrt(2) ~= 0.7071
    assert np.isclose(sine_rms, 1.0 / np.sqrt(2.0), atol=1e-2)

    sine_crest = crest_factor(sine)
    # Theoretical Crest Factor for pure sine is sqrt(2) ~= 1.4142
    assert np.isclose(sine_crest, np.sqrt(2.0), atol=2e-2)

    sine_kurt = kurtosis(sine)
    # Theoretical Kurtosis for pure sine is 1.5
    assert np.isclose(sine_kurt, 1.5, atol=2e-2)

    # Frequency estimation on clean single tone
    zcf = zero_crossing_frequency(sine, sample_rate)
    assert np.isclose(zcf, f0, atol=2.0)

    fpf = fft_peak_frequency(sine, sample_rate)
    assert np.isclose(fpf, f0, atol=5.0)

    # 2. Known impulse / transient signal
    impulse = np.zeros(n)
    impulse[n // 2] = 5.0
    impulse_cf = crest_factor(impulse)
    impulse_kurt = kurtosis(impulse)

    # An impulse should exhibit very high crest factor and very high kurtosis
    assert impulse_cf > 20.0
    assert impulse_kurt > 500.0

    # 3. Constant / silent signal
    constant = np.zeros(n)
    feats = extract_features(constant, sample_rate)
    assert feats["rms"] == 0.0
    assert feats["crest_factor"] == 0.0
    assert feats["kurtosis"] == 0.0
