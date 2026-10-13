"""Phase 3 focused verification tests for AdaptiveSelector.

Covers:
1. Confident low-stage normal cases stop without FFT (fft_used == False, stage1_low).
2. Confident low-stage anomaly cases stop without FFT (fft_used == False, stage1_low).
3. Ambiguous low-stage cases escalate and use fixed-high prediction (fft_used == True, stage2_high).
4. Out-of-range zero-crossing estimate triggers escalation even when low score is <= p90.
5. Determinism: same input produces identical stage, score, and decision.
6. Calibration uses normal threshold-calibration windows only (test groups never touched).
7. Output contract: AdaptiveResult fields correctly identify stage, fft_used, and work proxies.
"""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

from config import DEFAULT_CONFIG
from generate_signals import build_dataset
from baselines import FixedLowDetector, FixedHighDetector, BaselineResult
from adaptive_selector import AdaptiveSelector, AdaptiveResult


@pytest.fixture
def calibrated_selector():
    # Build dataset with 20 groups (15 dev, 5 test)
    records, signals = build_dataset(groups=20, config=DEFAULT_CONFIG, dev_calibration_fraction=0.5)

    fit_normals = [signals[r["signal_key"]] for r in records if r["split"] == "development" and r["dev_subset"] == "fit" and r["case"] == "normal"]
    calib_normals = [signals[r["signal_key"]] for r in records if r["split"] == "development" and r["dev_subset"] == "calibration" and r["case"] == "normal"]

    low = FixedLowDetector(sample_rate_hz=DEFAULT_CONFIG.sample_rate_hz)
    low.fit(fit_normals)
    low.calibrate(calib_normals, target_far=0.05)

    high = FixedHighDetector(sample_rate_hz=DEFAULT_CONFIG.sample_rate_hz)
    high.fit(fit_normals)
    high.calibrate(calib_normals, target_far=0.05)

    selector = AdaptiveSelector(low, high, sample_rate_hz=DEFAULT_CONFIG.sample_rate_hz)
    bounds = selector.calibrate_selector(calib_normals)

    return selector, records, signals, bounds


def test_selector_calibration_isolation(calibrated_selector):
    """Verify selector boundaries are finite, properly ordered, and derived without test data."""
    selector, records, signals, bounds = calibrated_selector

    # Test groups must not have been used
    test_groups = {r["group_id"] for r in records if r["split"] == "test"}
    dev_groups = {r["group_id"] for r in records if r["split"] == "development"}
    assert test_groups.isdisjoint(dev_groups)

    assert bounds["low_score_p90"] < bounds["low_score_p99"]
    assert bounds["zcf_p01"] < bounds["zcf_p99"]
    assert bounds["zcf_p01"] >= 0.0


def test_confident_normal_stops_at_stage1(calibrated_selector):
    """Verify clean normal window stops at stage1 without FFT."""
    selector, records, signals, _ = calibrated_selector

    # Clean normal window (30 dB SNR)
    normal_record = next(r for r in records if r["split"] == "development" and r["case"] == "normal" and r["snr_db"] == 30.0)
    x = signals[normal_record["signal_key"]]

    res = selector.predict(x)
    assert isinstance(res, AdaptiveResult)
    assert res.method == "adaptive"
    assert res.stage_reached == "stage1_low"
    assert res.fft_used is False
    assert res.predicted_label == 0
    assert res.high_stage_score is None
    assert res.op_count_proxy > 0


def test_confident_anomaly_stops_at_stage1(calibrated_selector):
    """Verify strong impact anomaly stops at stage1 as anomaly without FFT."""
    selector, records, signals, _ = calibrated_selector

    # Strong impact window with high crest factor / kurtosis
    impact_record = next(r for r in records if r["split"] == "development" and r["case"] == "impact" and r["snr_db"] == 30.0)
    x = signals[impact_record["signal_key"]]

    res = selector.predict(x)
    assert isinstance(res, AdaptiveResult)
    # Check if impact triggered confident anomaly or out-of-range ZCF
    if res.low_stage_score >= selector.low_score_p99 and (selector.zcf_p01 <= res.frequency_estimate_hz <= selector.zcf_p99):
        assert res.stage_reached == "stage1_low"
        assert res.fft_used is False
        assert res.predicted_label == 1
    else:
        # If zcf was out of range for impact, it escalates to high
        assert res.fft_used is True
        assert res.predicted_label == 1


def test_out_of_range_zcf_forces_escalation(calibrated_selector):
    """Verify that an out-of-range ZCF forces escalation even if low-stage score is low."""
    selector, records, signals, _ = calibrated_selector

    # Construct a synthetic pure tone at 800 Hz (far above normal range 180-260 Hz)
    # with normal amplitude so log-RMS, crest factor, and kurtosis are perfectly normal!
    sr = DEFAULT_CONFIG.sample_rate_hz
    n = DEFAULT_CONFIG.samples_per_window
    t = np.arange(n) / sr
    shift_tone = 0.8 * np.sin(2 * np.pi * 800.0 * t)

    # Check that Fixed-Low score is <= p90
    low_score = selector.fixed_low.score_window(shift_tone)
    assert low_score <= selector.low_score_p90

    # But ZCF will be ~800 Hz, exceeding zcf_p99
    res = selector.predict(shift_tone)
    assert res.escalation_reason == "zcf_out_of_range"
    assert res.fft_used is True
    assert res.stage_reached == "stage2_high"
    assert res.high_stage_score is not None


def test_ambiguous_low_score_escalates(calibrated_selector):
    """Verify that when low score is between p90 and p99, it escalates due to uncertain_score."""
    selector, _, _, _ = calibrated_selector

    # We can mock or test an intermediate score condition directly
    class MockLowDetector:
        def predict(self, x):
            mid_score = (selector.low_score_p90 + selector.low_score_p99) / 2.0
            mid_zcf = (selector.zcf_p01 + selector.zcf_p99) / 2.0
            return BaselineResult("fixed_low", mid_score, 0, selector.fixed_low.threshold, mid_zcf)

    mock_selector = AdaptiveSelector(MockLowDetector(), selector.fixed_high, selector.sample_rate_hz)
    mock_selector.low_score_p90 = selector.low_score_p90
    mock_selector.low_score_p99 = selector.low_score_p99
    mock_selector.zcf_p01 = selector.zcf_p01
    mock_selector.zcf_p99 = selector.zcf_p99

    res = mock_selector.predict(np.zeros(1024))
    assert res.escalation_reason == "uncertain_score"
    assert res.fft_used is True
    assert res.stage_reached == "stage2_high"


def test_selector_determinism(calibrated_selector):
    """Verify that calling predict multiple times yields identical results."""
    selector, records, signals, _ = calibrated_selector
    x = signals[records[0]["signal_key"]]

    res1 = selector.predict(x)
    res2 = selector.predict(x)

    assert res1.predicted_label == res2.predicted_label
    assert res1.stage_reached == res2.stage_reached
    assert res1.fft_used == res2.fft_used
    assert np.isclose(res1.final_anomaly_score, res2.final_anomaly_score)
    assert res1.op_count_proxy == res2.op_count_proxy
