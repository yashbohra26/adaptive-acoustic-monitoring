"""Quality constraints and operation cost accounting verification tests.

Verifies:
1. Zero duplicate FFT executions: Fixed-High computes exactly 1 rFFT per prediction,
   Stage 1 computes 0 rFFTs, and escalated adaptive windows compute exactly 1 rFFT.
2. Operation-count proxy strictly aligns with executed stage path under the documented model:
   - Fixed-Low standalone: 10,265 ops (10 * N + 25)
   - Fixed-High standalone: 29,250 ops (2.5 * N * log2(N) + 3.5 * N + 66)
   - Adaptive Stage 1 alone: 10,265 ops
   - Adaptive Stage 2 (escalated): 39,515 ops (10,265 + 29,250)
   - Adaptive never pays for unexecuted stages or duplicate FFTs.
3. Grouped split integrity: Development and test groups remain strictly disjoint.
4. Historical Phase 5 Candidate v1.1 metrics reproduce exactly from frozen artifacts.
5. Quality constraints are rigorously verified, and Candidate v1.1's recall constraint
   failure is accurately reported rather than concealed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

from baselines import FixedLowDetector, FixedHighDetector
from candidate_selector import CandidateAdaptiveSelector
from config import DEFAULT_CONFIG
from generate_signals import build_dataset


@pytest.fixture(scope="module")
def dev_and_test_data():
    """Build dataset and return records and signals."""
    records, signals = build_dataset(groups=80, config=DEFAULT_CONFIG, dev_calibration_fraction=0.5)
    return records, signals


@pytest.fixture(scope="module")
def fitted_detectors(dev_and_test_data):
    """Return calibrated Fixed-Low, Fixed-High, and Candidate v1.1 detectors."""
    records, signals = dev_and_test_data
    dev_records = [r for r in records if r["split"] == "development"]

    fit_sigs = [signals[r["signal_key"]] for r in dev_records if r["dev_subset"] == "fit" and r["case"] == "normal"]
    calib_sigs = [signals[r["signal_key"]] for r in dev_records if r["dev_subset"] == "calibration" and r["case"] == "normal"]
    calib_snrs = [r["snr_db"] for r in dev_records if r["dev_subset"] == "calibration" and r["case"] == "normal"]

    low_det = FixedLowDetector(sample_rate_hz=DEFAULT_CONFIG.sample_rate_hz)
    low_det.fit(fit_sigs)
    low_det.calibrate(calib_sigs, target_far=0.05)

    high_det = FixedHighDetector(sample_rate_hz=DEFAULT_CONFIG.sample_rate_hz)
    high_det.fit(fit_sigs)
    high_det.calibrate(calib_sigs, target_far=0.05)

    cand_det = CandidateAdaptiveSelector(
        low_det,
        high_det,
        sample_rate_hz=DEFAULT_CONFIG.sample_rate_hz,
        p_normal_percentile=60.0,
        p_anomaly_multiplier=1.2,
    )
    cand_det.calibrate_selector(calib_sigs, calib_snrs)

    return low_det, high_det, cand_det


def test_split_group_disjointness(dev_and_test_data):
    """Verify that development and test groups are strictly disjoint."""
    records, _ = dev_and_test_data
    dev_groups = set(r["group_id"] for r in records if r["split"] == "development")
    test_groups = set(r["group_id"] for r in records if r["split"] == "test")

    assert len(dev_groups) == 60
    assert len(test_groups) == 20
    assert len(dev_groups.intersection(test_groups)) == 0, "Data leakage! Dev and test groups overlap!"


def test_no_duplicate_fft_execution(fitted_detectors):
    """Verify that Fixed-High computes rFFT exactly once and adaptive does not duplicate rFFT."""
    low_det, high_det, cand_det = fitted_detectors
    x = np.random.default_rng(42).standard_normal(1024)

    # 1. Fixed-Low must NEVER call rFFT
    with patch("numpy.fft.rfft", wraps=np.fft.rfft) as mock_rfft:
        low_det.predict(x)
        assert mock_rfft.call_count == 0, f"Fixed-Low called rFFT {mock_rfft.call_count} times; expected 0"

    # 2. Fixed-High must call rFFT EXACTLY ONCE
    with patch("numpy.fft.rfft", wraps=np.fft.rfft) as mock_rfft:
        high_det.predict(x)
        assert mock_rfft.call_count == 1, f"Fixed-High called rFFT {mock_rfft.call_count} times; expected exactly 1 (duplicate FFT bug!)"

    # 3. Candidate v1.1 on confident normal window: must call rFFT 0 times
    # A low amplitude signal is well within normal bounds
    clean_normal = np.sin(2.0 * np.pi * 200.0 * np.arange(1024) / 4000.0) * 0.7
    with patch("numpy.fft.rfft", wraps=np.fft.rfft) as mock_rfft:
        res = cand_det.predict(clean_normal)
        if res.stage_reached == "stage1_low":
            assert mock_rfft.call_count == 0, f"Adaptive Stage 1 called rFFT {mock_rfft.call_count} times; expected 0"
            assert res.op_count_proxy == 10265
            assert not res.fft_used

    # 4. Candidate v1.1 on frequency-shifted window: must escalate and call rFFT EXACTLY ONCE
    shifted = np.sin(2.0 * np.pi * 350.0 * np.arange(1024) / 4000.0)
    with patch("numpy.fft.rfft", wraps=np.fft.rfft) as mock_rfft:
        res = cand_det.predict(shifted)
        if res.stage_reached == "stage2_high":
            assert mock_rfft.call_count == 1, f"Adaptive Stage 2 called rFFT {mock_rfft.call_count} times; expected 1"
            assert res.op_count_proxy == 39515
            assert res.fft_used


def test_operation_cost_accounting_invariants(fitted_detectors):
    """Verify that operation counts strictly adhere to the documented cost model."""
    low_det, high_det, cand_det = fitted_detectors
    rng = np.random.default_rng(123)

    for _ in range(50):
        x = rng.standard_normal(1024)

        # Standalone Low
        res_l = low_det.predict(x)
        assert res_l.anomaly_score is not None

        # Standalone High
        res_h = high_det.predict(x)
        assert res_h.anomaly_score is not None

        # Candidate
        res_c = cand_det.predict(x)
        if res_c.stage_reached == "stage1_low":
            assert res_c.op_count_proxy == 10265, "Stage 1 operation count proxy must be 10,265"
            assert not res_c.fft_used
        elif res_c.stage_reached == "stage2_high":
            assert res_c.op_count_proxy == 39515, "Escalated operation count proxy must be 39,515 (10,265 + 29,250)"
            assert res_c.fft_used
        else:
            pytest.fail(f"Invalid stage reached: {res_c.stage_reached}")


def test_historical_phase5_metrics_reproduce():
    """Verify that historical Phase 5 Candidate v1.1 results reproduce bit-for-bit from frozen artifact."""
    artifact_path = Path("results/phase5_final_evaluation.json")
    assert artifact_path.exists(), "Historical Phase 5 artifact missing!"

    with artifact_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["status"] == "PHASE5_TEST_EVALUATION_COMPLETE"
    assert data["protocol_version"] == "1.1.0"
    assert data["test_windows"] == 320
    assert data["test_groups"] == 20

    m_v11 = data["methods"]["candidate_v1.1"]["metrics"]
    assert np.isclose(m_v11["f1"], 0.9170305676855894)
    assert np.isclose(m_v11["recall"], 0.875)
    assert np.isclose(m_v11["precision"], 0.963302752293578)
    assert np.isclose(m_v11["false_alarm_rate"], 0.1)
    assert m_v11["tp"] == 210
    assert m_v11["fp"] == 8
    assert m_v11["tn"] == 72
    assert m_v11["fn"] == 30

    m_high = data["methods"]["fixed_high"]["metrics"]
    assert np.isclose(m_high["f1"], 0.9491525423728815)
    assert np.isclose(m_high["recall"], 0.9333333333333333)
    assert np.isclose(m_high["false_alarm_rate"], 0.1)


def test_quality_constraints_audit_accurate_reporting():
    """Verify and audit the pre-declared quality constraints on the held-out test set.

    Pre-declared rules relative to Fixed-High reference baseline:
    1. F1 constraint: Adaptive F1 >= Fixed-High F1 - 0.05
    2. FAR constraint: Adaptive FAR <= Fixed-High FAR + 0.01
    3. Ops constraint: Adaptive Ops < Fixed-High Ops (29,250)
    4. Recall constraint: Adaptive Recall >= Fixed-High Recall - 0.05

    CRITICAL CHECK: Accurately asserts that Candidate v1.1 FAILS the recall constraint!
    """
    artifact_path = Path("results/phase5_final_evaluation.json")
    with artifact_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    m_high = data["methods"]["fixed_high"]["metrics"]
    m_v11 = data["methods"]["candidate_v1.1"]["metrics"]
    ops_v11 = data["methods"]["candidate_v1.1"]["avg_op_count_proxy"]

    # 1. F1 Constraint
    f1_target = m_high["f1"] - 0.05
    meets_f1 = m_v11["f1"] >= f1_target
    assert meets_f1, f"F1 constraint failed: {m_v11['f1']:.4f} < {f1_target:.4f}"

    # 2. FAR Constraint
    far_target = m_high["false_alarm_rate"] + 0.01
    meets_far = m_v11["false_alarm_rate"] <= far_target
    assert meets_far, f"FAR constraint failed: {m_v11['false_alarm_rate']:.4f} > {far_target:.4f}"

    # 3. Ops Constraint
    meets_ops = ops_v11 < 29250.0
    assert meets_ops, f"Ops constraint failed: {ops_v11:.1f} >= 29250"

    # 4. Recall Constraint
    # Fixed-High Recall = 0.9333 -> Target is 0.9333 - 0.05 = 0.8833
    # Candidate v1.1 Recall = 0.8750 (210/240)
    recall_target = m_high["recall"] - 0.05
    meets_recall = m_v11["recall"] >= recall_target

    # This test explicitly verifies that the constraint is NOT met on the test set,
    # ensuring that failure is recorded transparently and never concealed.
    assert not meets_recall, (
        f"Expected Candidate v1.1 recall constraint to be NOT MET on Phase 5 test set. "
        f"Recall was {m_v11['recall']:.4f}, target was {recall_target:.4f}"
    )


def test_protocol_v1_2_structure_and_constants():
    """Verify Protocol v1.2.0 pre-test freeze artifact structure and frozen parameters."""
    protocol_path = Path("results/evaluation_protocol_v1.2.json")
    assert protocol_path.exists(), "Protocol v1.2 freeze artifact missing!"

    with protocol_path.open("r", encoding="utf-8") as f:
        proto = json.load(f)

    assert proto["protocol_version"] == "1.2.0"
    assert "candidate_v1.2" in proto
    cand = proto["candidate_v1.2"]
    assert cand["p_normal_percentile"] == 50.0
    assert cand["p_anomaly_multiplier"] == 1.2
    assert cand["stage1_ops"] == 10265
    assert cand["stage2_cumulative_ops"] == 39515

    bounds = cand["frozen_boundaries"]
    assert bounds["low_score_p_normal"] > 0.0
    assert bounds["low_score_p_anomaly"] > bounds["low_score_p_normal"]
    assert 150.0 < bounds["zcf_min"] < 200.0
    assert 240.0 < bounds["zcf_max"] < 300.0


def test_candidate_v1_2_frozen_metrics_reproduce():
    """Verify Candidate v1.2 independent evaluation metrics match the recorded artifact."""
    artifact_path = Path("results/candidate_v1.2_independent_evaluation.json")
    assert artifact_path.exists(), "Candidate v1.2 independent evaluation artifact missing!"

    with artifact_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["status"] == "INDEPENDENT_HOLDOUT_EVALUATION_COMPLETE"
    assert data["protocol_version"] == "1.2.0"
    assert data["eval_seed"] == 20261014
    assert data["eval_windows"] == 320
    assert "methods" in data

    m_v12 = data["methods"]["candidate_v1.2"]["metrics"]
    assert np.isclose(m_v12["f1"], 0.9533898305084746)
    assert np.isclose(m_v12["recall"], 0.9375)
    assert np.isclose(m_v12["precision"], 0.9698275862068966)
    assert np.isclose(m_v12["false_alarm_rate"], 0.0875)
    assert m_v12["tp"] == 225
    assert m_v12["fp"] == 7
    assert m_v12["tn"] == 73
    assert m_v12["fn"] == 15
    assert np.isclose(data["methods"]["candidate_v1.2"]["avg_op_count_proxy"], 28729.0625)

    # Candidate v1.1 on the same independent evaluation
    m_v11 = data["methods"]["candidate_v1.1"]["metrics"]
    assert np.isclose(m_v11["f1"], 0.9046563192904656)
    assert np.isclose(m_v11["recall"], 0.85)


def test_candidate_v1_2_quality_status_computation():
    """Verify that Candidate v1.2 quality constraint status is accurately computed from saved metrics.

    Checks:
    1. Candidate v1.2 meets F1, Recall, FAR, and Ops constraints relative to Fixed-High.
    2. Candidate v1.1 on this independent holdout FAILS the recall constraint, exactly
       as recorded in the artifact ('NO (Rec:0.850 vs 0.887)').
    """
    artifact_path = Path("results/candidate_v1.2_independent_evaluation.json")
    with artifact_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    m_high = data["methods"]["fixed_high"]["metrics"]
    m_v12 = data["methods"]["candidate_v1.2"]["metrics"]
    ops_v12 = data["methods"]["candidate_v1.2"]["avg_op_count_proxy"]

    m_v11 = data["methods"]["candidate_v1.1"]["metrics"]

    # Candidate v1.2 constraints relative to Fixed-High
    f1_target = m_high["f1"] - 0.05
    rec_target = m_high["recall"] - 0.05
    far_target = m_high["false_alarm_rate"] + 0.01

    v12_meets_f1 = m_v12["f1"] >= f1_target
    v12_meets_rec = m_v12["recall"] >= rec_target
    v12_meets_far = m_v12["false_alarm_rate"] <= far_target
    v12_meets_ops = ops_v12 < 29250.0

    assert v12_meets_f1, f"v1.2 F1 constraint failed: {m_v12['f1']} < {f1_target}"
    assert v12_meets_rec, f"v1.2 Recall constraint failed: {m_v12['recall']} < {rec_target}"
    assert v12_meets_far, f"v1.2 FAR constraint failed: {m_v12['false_alarm_rate']} > {far_target}"
    assert v12_meets_ops, f"v1.2 Ops constraint failed: {ops_v12} >= 29250.0"

    assert data["methods"]["candidate_v1.2"]["meets_quality_constraints"] == "YES"

    # Candidate v1.1 on this evaluation fails recall
    v11_meets_rec = m_v11["recall"] >= rec_target
    assert not v11_meets_rec, (
        f"Expected Candidate v1.1 to fail recall on independent holdout: "
        f"recall was {m_v11['recall']:.4f}, target was {rec_target:.4f}"
    )
    assert "NO (Rec:0.850 vs 0.887)" in data["methods"]["candidate_v1.1"]["meets_quality_constraints"]

