"""Phase 5 verification tests for the final held-out test evaluation."""
import json
from pathlib import Path
import numpy as np
import pytest


def test_phase5_artifacts_exist_and_consistent():
    """Verify that phase 5 final evaluation outputs exist and match expected structure."""
    res_path = Path("results/phase5_final_evaluation.json")
    pred_path = Path("results/phase5_window_predictions.csv")

    assert res_path.exists(), "phase5_final_evaluation.json must exist"
    assert pred_path.exists(), "phase5_window_predictions.csv must exist"

    with res_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["status"] == "PHASE5_TEST_EVALUATION_COMPLETE"
    assert data["test_windows"] == 320
    assert data["test_groups"] == 20
    assert data["test_split_hash_sha256"].startswith("b99e0b36ea5b6659")

    methods = data["methods"]
    for m in ["fixed_low", "fixed_high", "adaptive_v1.0", "candidate_v1.1"]:
        assert m in methods
        metrics = methods[m]["metrics"]
        assert 0.0 <= metrics["precision"] <= 1.0
        assert 0.0 <= metrics["recall"] <= 1.0
        assert 0.0 <= metrics["f1"] <= 1.0
        assert 0.0 <= metrics["false_alarm_rate"] <= 1.0


def test_phase5_bootstrap_cis_valid():
    """Verify group-level bootstrap confidence intervals bracket point estimates."""
    res_path = Path("results/phase5_final_evaluation.json")
    with res_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    cis = data["bootstrap_95ci"]
    methods = data["methods"]

    for m in ["fixed_low", "fixed_high", "adaptive_v1.0", "candidate_v1.1"]:
        ci = cis[m]
        point = methods[m]["metrics"]

        # Check F1 and recall intervals
        assert ci["f1"][0] <= point["f1"] <= ci["f1"][1] or np.isclose(ci["f1"][0], point["f1"]) or np.isclose(ci["f1"][1], point["f1"])
        assert ci["recall"][0] <= point["recall"] <= ci["recall"][1] or np.isclose(ci["recall"][0], point["recall"]) or np.isclose(ci["recall"][1], point["recall"])


def test_phase5_work_accounting():
    """Verify that Candidate v1.1 achieved positive operation savings over Fixed-High."""
    res_path = Path("results/phase5_final_evaluation.json")
    with res_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    high_ops = data["methods"]["fixed_high"]["avg_op_count_proxy"]
    cand_ops = data["methods"]["candidate_v1.1"]["avg_op_count_proxy"]
    savings = data["methods"]["candidate_v1.1"]["savings_vs_fixed_high_percent"]

    assert high_ops == 29250.0
    assert cand_ops < high_ops
    assert savings > 0.0
    expected_savings = (high_ops - cand_ops) / high_ops * 100.0
    assert np.isclose(savings, expected_savings)
