"""Tests for Phase 4 feature investigation, identifiability audit, and Candidate Selector v1.1."""
import json
from pathlib import Path
import numpy as np
import pytest

from baselines import FixedLowDetector, FixedHighDetector
from candidate_features import normalized_difference_ratio, get_clean_calibrated_zcf_bounds
from candidate_selector import CandidateAdaptiveSelector, CandidateResult


def test_normalized_difference_ratio_pure_sinusoid():
    """Verify difference ratio formula matches expectation for known frequency."""
    fs = 4000
    f = 200.0
    n = 1024
    t = np.arange(n) / fs
    x = np.sin(2 * np.pi * f * t)
    dr = normalized_difference_ratio(x)
    expected = 2.0 * (1.0 - np.cos(2 * np.pi * f / fs))
    assert abs(dr - expected) < 0.01


def test_normalized_difference_ratio_noise():
    """Verify white noise diff ratio is close to 2.0 (uncorrelated)."""
    rng = np.random.default_rng(42)
    noise = rng.normal(0.0, 1.0, 4000)
    dr = normalized_difference_ratio(noise)
    assert 1.90 < dr < 2.10


def test_clean_calibrated_zcf_bounds():
    """Verify clean calibration ZCF bounds filter by SNR."""
    fs = 4000
    t = np.arange(1024) / fs
    clean_200 = np.sin(2 * np.pi * 200.0 * t)
    clean_250 = np.sin(2 * np.pi * 250.0 * t)
    noisy_0db = clean_200 + np.random.default_rng(0).normal(0.0, 1.0, 1024)

    sigs = [clean_200, clean_250, noisy_0db]
    snrs = [30.0, 20.0, 0.0]

    z_min, z_max = get_clean_calibrated_zcf_bounds(sigs, snrs, sample_rate_hz=fs, margin_hz=10.0)
    assert 185.0 < z_min < 195.0  # 200 - 10
    assert 255.0 < z_max < 265.0  # 250 + 10


def test_candidate_selector_no_leakage_and_correct_stages():
    """Verify candidate selector does not require or accept ground-truth metadata."""
    fs = 4000
    t = np.arange(1024) / fs
    normal_sigs = [np.sin(2 * np.pi * 200.0 * t + np.random.default_rng(i).uniform(0, 2*np.pi)) for i in range(20)]
    normal_snrs = [30.0, 20.0, 10.0, 0.0] * 5

    low = FixedLowDetector(sample_rate_hz=fs)
    low.fit(normal_sigs[:10])
    low.calibrate(normal_sigs[10:], target_far=0.05)

    high = FixedHighDetector(sample_rate_hz=fs)
    high.fit(normal_sigs[:10])
    high.calibrate(normal_sigs[10:], target_far=0.05)

    cand = CandidateAdaptiveSelector(low, high, sample_rate_hz=fs, p_normal_percentile=60.0, p_anomaly_multiplier=1.2)
    bounds = cand.calibrate_selector(normal_sigs[10:], normal_snrs[10:])

    assert "low_score_p_normal" in bounds
    assert "low_score_p_anomaly" in bounds
    assert "zcf_min" in bounds
    assert "zcf_max" in bounds

    # Normal window
    res_norm = cand.predict(normal_sigs[0])
    assert isinstance(res_norm, CandidateResult)
    assert res_norm.predicted_label in (0, 1)

    # Obvious impact window: should stop at stage1 without FFT
    impact_win = np.sin(2 * np.pi * 900.0 * t) * np.exp(-t / 0.005)
    res_imp = cand.predict(impact_win)
    assert res_imp.stage_reached == "stage1_low"
    assert res_imp.fft_used is False
    assert res_imp.escalation_reason == "confident_anomaly"
    assert res_imp.op_count_proxy == 10 * 1024 + 25

    # Frequency shifted window: should escalate to stage2
    shift_win = np.sin(2 * np.pi * 350.0 * t)
    res_shift = cand.predict(shift_win)
    assert res_shift.stage_reached == "stage2_high"
    assert res_shift.fft_used is True
    assert res_shift.escalation_reason == "zcf_out_of_range"
    assert res_shift.op_count_proxy == (10 * 1024 + 25) + int(2.5 * 1024 * 10 + 3.5 * 1024 + 66)


def test_identifiability_audit_artifact_exists():
    """Verify identifiability audit artifact exists and has correct break-even arithmetic."""
    audit_path = Path("results/phase4_study/identifiability_audit.json")
    assert audit_path.exists(), "identifiability_audit.json must exist"

    with audit_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    cost_info = data["cascade_cost_accounting"]
    assert cost_info["stage1_ops"] == 10265
    assert cost_info["stage2_added_ops"] == 29250
    assert cost_info["escalated_total_ops"] == 39515
    assert cost_info["standalone_fixed_high_ops"] == 29250
    assert abs(cost_info["break_even_escalation_rate"] - (29250 - 10265) / 29250) < 1e-4

    # Inputs audit: must explicitly show metadata is NOT available at inference time
    inputs = data["inference_time_inputs_available"]
    assert inputs["window_samples_x"] is True
    assert inputs["source_group_id"] is False
    assert inputs["nominal_normal_frequency_hz"] is False
    assert inputs["ground_truth_case"] is False
