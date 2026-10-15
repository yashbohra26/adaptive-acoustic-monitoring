"""Phase 5 verification tests for the final held-out test evaluation."""
import json
import csv
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

import run_phase5_evaluation as phase5
from generate_signals import assert_disjoint_source_group_splits


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
    # Historical artifacts intentionally retain their original schema and hashes.
    with pred_path.open("r", encoding="utf-8", newline="") as f:
        historical_columns = set(next(csv.reader(f)))
    assert "v10_freq_hz" not in historical_columns
    assert "v11_freq_hz" not in historical_columns

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


def test_frequency_metric_counts_eligible_and_scores_all_expected_estimates():
    records = [
        {"case": "normal", "target_frequency_hz": 100.0},
        {"case": "frequency_shift", "target_frequency_hz": 200.0},
        {"case": "added_harmonic", "target_frequency_hz": ""},
        {"case": "impact", "target_frequency_hz": None},
    ]
    result = phase5.compute_frequency_metrics(records, [101.0, 197.0, None, None])
    assert result == {
        "mae_hz": 2.0,
        "eligible_count": 2,
        "scored_count": 2,
        "missing_prediction_count": 0,
    }


def test_frequency_metric_rejects_missing_prediction_for_eligible_window():
    records = [
        {"target_frequency_hz": 100.0},
        {"target_frequency_hz": 200.0},
        {"target_frequency_hz": ""},
    ]
    with pytest.raises(ValueError, match="1 of 2 eligible windows"):
        phase5.compute_frequency_metrics(records, [99.0, None, None])


def test_frequency_metric_rejects_misaligned_predictions():
    with pytest.raises(ValueError, match="2 predictions for 1 records"):
        phase5.compute_frequency_metrics([{"target_frequency_hz": 100.0}], [99.0, 100.0])


def test_source_group_split_guard_rejects_group_in_both_partitions():
    records = [
        {"group_id": "g0001", "split": "development"},
        {"group_id": "g0001", "split": "test"},
    ]
    with pytest.raises(AssertionError, match="Source-group leakage.*g0001"):
        assert_disjoint_source_group_splits(records)


def test_future_run_provenance_records_explicit_git_unavailable_fallback(monkeypatch, tmp_path):
    def git_unavailable(*args, **kwargs):
        raise FileNotFoundError("git executable unavailable")

    monkeypatch.setattr(phase5.subprocess, "run", git_unavailable)
    provenance = phase5.collect_run_provenance(tmp_path)
    assert provenance["git_revision"] == {
        "commit": None,
        "status": "unavailable",
        "error": "FileNotFoundError: git executable unavailable",
    }
    assert provenance["git_working_tree"]["dirty"] is None
    assert provenance["git_working_tree"]["status"] == "unavailable"
    assert provenance["python_environment"]["status"] == "available"
    assert provenance["python_environment"]["python_version"]
    assert provenance["python_environment"]["numpy_version"] == np.__version__
    assert phase5.SUMMARY_ARTIFACT_V2 != "phase5_final_evaluation.json"
    assert phase5.PREDICTIONS_ARTIFACT_V2 != "phase5_window_predictions.csv"


def test_future_run_provenance_records_environment_unavailable_fallback(monkeypatch, tmp_path):
    monkeypatch.setattr(
        phase5.subprocess,
        "run",
        lambda *args, **kwargs: phase5.subprocess.CompletedProcess(
            args, 0, stdout="", stderr=""
        ),
    )

    def environment_unavailable():
        raise RuntimeError("environment probe unavailable")

    monkeypatch.setattr(phase5.platform, "python_version", environment_unavailable)
    provenance = phase5.collect_run_provenance(tmp_path)
    assert provenance["python_environment"] == {
        "status": "unavailable",
        "python_version": None,
        "implementation": None,
        "platform": None,
        "numpy_version": None,
        "error": "RuntimeError: environment probe unavailable",
    }


def test_fair_benchmark_times_all_four_methods_with_one_protocol():
    """Historical Phase 5 placeholders are superseded by the uniform timing artifact."""
    path = Path("results/benchmark_fair_evaluation.json")
    assert path.exists(), "Fair four-method runtime benchmark artifact is missing"
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    timing = data["timing_parameters"]
    assert timing["warmup_iterations"] == 5
    assert timing["timed_repetitions"] == 25
    assert timing["timer"] == "time.perf_counter_ns"
    assert set(data["methods"]) == {
        "fixed_low", "fixed_high", "candidate_v1.1", "historical_v1.0"
    }
    for name, method in data["methods"].items():
        runtime = method["runtime_ms"]
        assert runtime["mean"] > 0.0, f"{name} runtime is missing or a zero placeholder"
        assert runtime["std"] >= 0.0


def test_benchmark_operation_counts_match_executed_stages():
    """Reconcile per-window stage flags, cost proxy, and all-method summaries."""
    with Path("results/benchmark_fair_windows.csv").open(
        "r", encoding="utf-8", newline=""
    ) as f:
        rows = list(csv.DictReader(f))
    with Path("results/benchmark_fair_evaluation.json").open("r", encoding="utf-8") as f:
        summary = json.load(f)

    assert len(rows) == summary["test_windows"] == 320
    for method in summary["methods"]:
        recorded = []
        for row in rows:
            fft_used = row[f"{method}_fft"].lower() == "true"
            ops = int(row[f"{method}_ops"])
            expected = 29250 if method == "fixed_high" else 39515 if fft_used else 10265
            assert ops == expected, f"{method} cost {ops} contradicts its executed FFT/stage flag"
            assert fft_used is (method == "fixed_high" or row[f"{method}_stage"] == "stage2_high")
            recorded.append(ops)
        assert np.isclose(
            np.mean(recorded), summary["methods"][method]["avg_op_count_proxy"]
        ), f"{method} aggregate cost does not match its window records"
