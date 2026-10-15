"""Phase 5: Single frozen final evaluation on the 20 held-out test source groups.

Evaluates:
1. Fixed-Low baseline
2. Fixed-High baseline
3. Historical Adaptive Selector v1.0
4. Candidate Adaptive Selector v1.1 (Protocol v1.1.0)

Rules:
- Models are fit and calibrated EXCLUSIVELY on development partitions (60 groups).
- Test data (20 groups, 320 windows) is evaluated ONCE using frozen parameters.
- No tuning, threshold modification, or rule alteration is performed.
- Group-level bootstrap resampling (1,000 iterations, seed 20261013) constructs 95% CIs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np

from config import DEFAULT_CONFIG
from generate_signals import build_dataset
from baselines import FixedLowDetector, FixedHighDetector
from adaptive_selector import AdaptiveSelector
from candidate_selector import CandidateAdaptiveSelector


SUMMARY_ARTIFACT_V2 = "phase5_final_evaluation_v2.json"
PREDICTIONS_ARTIFACT_V2 = "phase5_window_predictions_v2.csv"


def compute_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, float]:
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    far = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_alarm_rate": far,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def compute_frequency_metrics(
    records: list[dict], freq_ests: list[float | None]
) -> dict[str, float | int | None]:
    """Compute MAE over target-defined windows and reject missing eligible estimates.

    A blank/None target means frequency MAE is legitimately ineligible (multi-tone
    or transient cases). A missing estimate on a target-defined window is an
    evaluation integrity error and must not silently shrink the MAE denominator.
    """
    if len(records) != len(freq_ests):
        raise ValueError(
            "Frequency predictions must align one-to-one with records: "
            f"{len(freq_ests)} predictions for {len(records)} records"
        )

    eligible = [
        (record, estimate)
        for record, estimate in zip(records, freq_ests)
        if record.get("target_frequency_hz") not in (None, "")
    ]
    errors: list[float] = []
    missing_count = 0
    for record, estimate in eligible:
        if estimate is None or not np.isfinite(float(estimate)):
            missing_count += 1
            continue
        target = float(record["target_frequency_hz"])
        errors.append(abs(float(estimate) - target))

    if missing_count:
        raise ValueError(
            "Frequency MAE has missing/non-finite predictions for "
            f"{missing_count} of {len(eligible)} eligible windows; "
            "refusing to report a potentially biased MAE"
        )

    return {
        "mae_hz": float(np.mean(errors)) if errors else None,
        "eligible_count": len(eligible),
        "scored_count": len(errors),
        "missing_prediction_count": missing_count,
    }


def collect_run_provenance(project_root: Path | None = None) -> dict:
    """Capture best-effort local code/environment provenance without network access."""
    root = project_root or Path(__file__).resolve().parents[1]
    revision: str | None = None
    revision_status = "unavailable"
    revision_error: str | None = None
    dirty: bool | None = None
    dirty_status = "unavailable"
    dirty_error: str | None = None

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
            text=True, check=True, timeout=3,
        )
        revision = result.stdout.strip() or None
        revision_status = "available" if revision else "unavailable"
        if revision is None:
            revision_error = "git rev-parse returned an empty revision"
    except (OSError, subprocess.SubprocessError) as exc:
        revision_error = f"{type(exc).__name__}: {exc}"

    try:
        result = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=root, capture_output=True, text=True, check=True, timeout=3,
        )
        dirty = bool(result.stdout.strip())
        dirty_status = "available"
    except (OSError, subprocess.SubprocessError) as exc:
        dirty_error = f"{type(exc).__name__}: {exc}"

    try:
        python_environment = {
            "status": "available",
            "python_version": platform.python_version(),
            "implementation": sys.implementation.name,
            "platform": platform.platform(),
            "numpy_version": np.__version__,
            "error": None,
        }
    except Exception as exc:  # best-effort provenance must not invent missing values
        python_environment = {
            "status": "unavailable",
            "python_version": None,
            "implementation": None,
            "platform": None,
            "numpy_version": None,
            "error": f"{type(exc).__name__}: {exc}",
        }

    return {
        "git_revision": {
            "commit": revision,
            "status": revision_status,
            "error": revision_error,
        },
        "git_working_tree": {
            "dirty": dirty,
            "status": dirty_status,
            "error": dirty_error,
        },
        "python_environment": python_environment,
    }


def assert_disjoint_source_group_splits(records: list[dict]) -> None:
    """Guard the evaluation runner against development/test group leakage."""
    from generate_signals import assert_disjoint_source_group_splits as check_splits

    check_splits(records)


def run_group_bootstrap(
    records: list[dict],
    predictions: dict[str, list[dict]],
    n_bootstraps: int = 1000,
    seed: int = 20261013,
) -> dict[str, dict[str, list[float]]]:
    """Group-level bootstrap resampling across test source groups."""
    groups = sorted(list(set(r["group_id"] for r in records)))
    group_indices = {g: [i for i, r in enumerate(records) if r["group_id"] == g] for g in groups}
    rng = np.random.default_rng(seed)

    methods = list(predictions.keys())
    boot_stats = {m: defaultdict(list) for m in methods}

    for _ in range(n_bootstraps):
        resampled_groups = rng.choice(groups, size=len(groups), replace=True)
        boot_idx = []
        for g in resampled_groups:
            boot_idx.extend(group_indices[g])

        boot_recs = [records[i] for i in boot_idx]
        y_true = [boot_recs[i]["anomaly"] for i in range(len(boot_recs))]

        for m in methods:
            preds_m = [predictions[m][i] for i in boot_idx]
            yp = [p["pred"] for p in preds_m]
            freqs = [p["freq"] for p in preds_m]
            ops = [p["ops"] for p in preds_m]
            ffts = [p["fft_used"] for p in preds_m]

            m_metrics = compute_metrics(y_true, yp)
            fmae = compute_frequency_metrics(boot_recs, freqs)["mae_hz"]
            if fmae is None:
                raise ValueError("Bootstrap replicate contains no frequency-eligible windows")

            boot_stats[m]["f1"].append(m_metrics["f1"])
            boot_stats[m]["recall"].append(m_metrics["recall"])
            boot_stats[m]["precision"].append(m_metrics["precision"])
            boot_stats[m]["false_alarm_rate"].append(m_metrics["false_alarm_rate"])
            boot_stats[m]["freq_mae_hz"].append(fmae)
            boot_stats[m]["avg_ops"].append(float(np.mean(ops)))
            boot_stats[m]["escalation_rate"].append(float(np.mean(ffts)))

    ci_results = {}
    for m in methods:
        ci_results[m] = {}
        for metric, values in boot_stats[m].items():
            ci_results[m][metric] = [
                float(np.percentile(values, 2.5)),
                float(np.percentile(values, 97.5)),
            ]

    return ci_results


def run_phase5_evaluation(out_dir: Path = Path("results")) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = out_dir / SUMMARY_ARTIFACT_V2
    predictions_path = out_dir / PREDICTIONS_ARTIFACT_V2
    existing_outputs = [path.name for path in (summary_path, predictions_path) if path.exists()]
    if existing_outputs:
        raise FileExistsError(
            "Refusing to overwrite versioned future-run artifacts: "
            + ", ".join(existing_outputs)
        )

    print("Building full dataset (80 source groups)...")
    records, signals = build_dataset(groups=80, config=DEFAULT_CONFIG, dev_calibration_fraction=0.5)

    dev_records = [r for r in records if r["split"] == "development"]
    test_records = [r for r in records if r["split"] == "test"]
    assert_disjoint_source_group_splits(records)
    assert len(dev_records) == 960
    assert len(test_records) == 320

    # Verify test split hash integrity
    test_groups = sorted(list(set(r["group_id"] for r in test_records)))
    test_hash = hashlib.sha256(",".join(test_groups).encode("utf-8")).hexdigest()
    assert test_hash.startswith("b99e0b36ea5b6659"), f"Hash mismatch: {test_hash[:16]}"
    print(f"Verified held-out test split: 320 windows across 20 groups (SHA-256: {test_hash[:16]}).")

    # Fit and calibrate models strictly on development data
    fit_sigs = [signals[r["signal_key"]] for r in dev_records if r["dev_subset"] == "fit" and r["case"] == "normal"]
    calib_sigs = [signals[r["signal_key"]] for r in dev_records if r["dev_subset"] == "calibration" and r["case"] == "normal"]
    calib_recs = [r for r in dev_records if r["dev_subset"] == "calibration" and r["case"] == "normal"]
    calib_snrs = [r["snr_db"] for r in calib_recs]

    low_det = FixedLowDetector(sample_rate_hz=DEFAULT_CONFIG.sample_rate_hz)
    low_det.fit(fit_sigs)
    low_det.calibrate(calib_sigs, target_far=0.05)

    high_det = FixedHighDetector(sample_rate_hz=DEFAULT_CONFIG.sample_rate_hz)
    high_det.fit(fit_sigs)
    high_det.calibrate(calib_sigs, target_far=0.05)

    v10_selector = AdaptiveSelector(low_det, high_det, sample_rate_hz=DEFAULT_CONFIG.sample_rate_hz)
    v10_selector.calibrate_selector(calib_sigs)

    v11_selector = CandidateAdaptiveSelector(
        low_det,
        high_det,
        sample_rate_hz=DEFAULT_CONFIG.sample_rate_hz,
        p_normal_percentile=60.0,
        p_anomaly_multiplier=1.2,
    )
    v11_selector.calibrate_selector(calib_sigs, calib_snrs)

    # Check boundaries match frozen protocol v1.1.0
    protocol_path = out_dir / "evaluation_protocol_v1.1.json"
    if protocol_path.exists():
        with protocol_path.open("r", encoding="utf-8") as f:
            p_spec = json.load(f)
            v11_spec = p_spec["frozen_methods"]["adaptive_v1.1_candidate"]["frozen_boundaries"]
            assert np.isclose(v11_selector.low_score_p_normal, v11_spec["low_score_p_normal"])
            assert np.isclose(v11_selector.low_score_p_anomaly, v11_spec["low_score_p_anomaly"])
            assert np.isclose(v11_selector.zcf_min, v11_spec["zcf_min_hz"])
            assert np.isclose(v11_selector.zcf_max, v11_spec["zcf_max_hz"])
        print("Verified all detector parameters match frozen Protocol v1.1.0.")

    # Execute single frozen evaluation on the 20 held-out groups (320 windows)
    print("\nExecuting single frozen evaluation on 320 held-out test windows...")
    predictions = {
        "fixed_low": [],
        "fixed_high": [],
        "adaptive_v1.0": [],
        "candidate_v1.1": [],
    }

    test_predictions_table = []
    y_true = [r["anomaly"] for r in test_records]

    for r in test_records:
        key = r["signal_key"]
        x = signals[key]

        # Fixed-Low
        res_l = low_det.predict(x)
        # Fixed-High
        res_h = high_det.predict(x)
        # Adaptive v1.0
        res_v10 = v10_selector.predict(x)
        # Candidate v1.1
        res_v11 = v11_selector.predict(x)

        predictions["fixed_low"].append({
            "pred": res_l.predicted_label,
            "score": res_l.anomaly_score,
            "freq": res_l.frequency_estimate_hz,
            "stage": "stage1_low",
            "fft_used": False,
            "ops": 10265,
            "reason": "none",
        })

        predictions["fixed_high"].append({
            "pred": res_h.predicted_label,
            "score": res_h.anomaly_score,
            "freq": res_h.frequency_estimate_hz,
            "stage": "stage2_high",
            "fft_used": True,
            "ops": 29250,
            "reason": "none",
        })

        predictions["adaptive_v1.0"].append({
            "pred": res_v10.predicted_label,
            "score": res_v10.final_anomaly_score,
            "freq": res_v10.frequency_estimate_hz,
            "stage": res_v10.stage_reached,
            "fft_used": res_v10.fft_used,
            "ops": res_v10.op_count_proxy,
            "reason": res_v10.escalation_reason,
        })

        predictions["candidate_v1.1"].append({
            "pred": res_v11.predicted_label,
            "score": res_v11.final_anomaly_score,
            "freq": res_v11.frequency_estimate_hz,
            "stage": res_v11.stage_reached,
            "fft_used": res_v11.fft_used,
            "ops": res_v11.op_count_proxy,
            "reason": res_v11.escalation_reason,
        })

        test_predictions_table.append({
            "group_id": r["group_id"],
            "case": r["case"],
            "snr_db": r["snr_db"],
            "anomaly_true": r["anomaly"],
            "target_freq_hz": r["target_frequency_hz"],
            "low_pred": res_l.predicted_label,
            "low_freq_hz": res_l.frequency_estimate_hz,
            "high_pred": res_h.predicted_label,
            "high_freq_hz": res_h.frequency_estimate_hz,
            "v10_freq_hz": res_v10.frequency_estimate_hz,
            "v10_pred": res_v10.predicted_label,
            "v10_stage": res_v10.stage_reached,
            "v10_fft": res_v10.fft_used,
            "v10_reason": res_v10.escalation_reason,
            "v10_ops": res_v10.op_count_proxy,
            "v11_pred": res_v11.predicted_label,
            "v11_stage": res_v11.stage_reached,
            "v11_fft": res_v11.fft_used,
            "v11_reason": res_v11.escalation_reason,
            "v11_ops": res_v11.op_count_proxy,
            "v11_freq_hz": res_v11.frequency_estimate_hz,
        })

    # New schema is written to distinct v2 filenames; historical artifacts are read-only.
    with predictions_path.open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(test_predictions_table[0].keys()))
        writer.writeheader()
        writer.writerows(test_predictions_table)

    # Compute overall metrics
    evaluation_summary = {
        "artifact_schema_version": "2.0",
        "status": "PHASE5_TEST_EVALUATION_COMPLETE",
        "protocol_version": "1.1.0",
        "test_windows": len(test_records),
        "test_groups": len(set(r["group_id"] for r in test_records)),
        "test_split_hash_sha256": test_hash,
        "provenance": collect_run_provenance(),
        "prediction_artifact": PREDICTIONS_ARTIFACT_V2,
        "runtime_measurement": {
            "status": "NOT_MEASURED_IN_FROZEN_PHASE5_EVALUATION",
            "note": "Use results/benchmark_fair_evaluation.json for desktop runtime, where all four methods use the same warmup and repeated timing harness.",
        },
        "methods": {},
    }

    print("\n--- Held-Out Test Evaluation Results (20 Groups, 320 Windows) ---")
    print(f"{'Method':<18} | {'F1':<6} | {'Recall':<6} | {'FAR':<6} | {'Freq MAE':<8} | {'Esc Rate':<8} | {'Avg Ops':<8} | {'Savings vs High':<15}")
    print("-" * 95)

    for m in ["fixed_low", "fixed_high", "adaptive_v1.0", "candidate_v1.1"]:
        yp = [p["pred"] for p in predictions[m]]
        freqs = [p["freq"] for p in predictions[m]]
        ops_list = [p["ops"] for p in predictions[m]]
        fft_list = [p["fft_used"] for p in predictions[m]]

        m_metrics = compute_metrics(y_true, yp)
        frequency_metrics = compute_frequency_metrics(test_records, freqs)
        fmae = frequency_metrics["mae_hz"]
        avg_ops = float(np.mean(ops_list))
        esc_rate = float(np.mean(fft_list))
        savings = (29250.0 - avg_ops) / 29250.0 * 100.0

        # Performance by case
        by_case = {}
        for c in ["normal", "frequency_shift", "added_harmonic", "impact"]:
            c_idx = [i for i, r in enumerate(test_records) if r["case"] == c]
            c_yt = [y_true[i] for i in c_idx]
            c_yp = [yp[i] for i in c_idx]
            cm = compute_metrics(c_yt, c_yp)
            c_esc = float(np.mean([fft_list[i] for i in c_idx]))
            c_ops = float(np.mean([ops_list[i] for i in c_idx]))
            by_case[c] = {
                "metric_rate": cm["false_alarm_rate"] if c == "normal" else cm["recall"],
                "tp": cm["tp"],
                "fp": cm["fp"],
                "tn": cm["tn"],
                "fn": cm["fn"],
                "escalation_rate": c_esc,
                "avg_ops": c_ops,
            }

        # Performance by SNR
        by_snr = {}
        for s in [30.0, 20.0, 10.0, 0.0]:
            s_idx = [i for i, r in enumerate(test_records) if r["snr_db"] == s]
            s_yt = [y_true[i] for i in s_idx]
            s_yp = [yp[i] for i in s_idx]
            sm = compute_metrics(s_yt, s_yp)
            s_esc = float(np.mean([fft_list[i] for i in s_idx]))
            s_ops = float(np.mean([ops_list[i] for i in s_idx]))
            by_snr[str(s)] = {
                "f1": sm["f1"],
                "recall": sm["recall"],
                "far": sm["false_alarm_rate"],
                "escalation_rate": s_esc,
                "avg_ops": s_ops,
            }

        evaluation_summary["methods"][m] = {
            "metrics": m_metrics,
            "freq_mae_hz": fmae,
            "frequency_metric_counts": {
                "eligible_count": frequency_metrics["eligible_count"],
                "scored_count": frequency_metrics["scored_count"],
                "missing_prediction_count": frequency_metrics["missing_prediction_count"],
            },
            "avg_op_count_proxy": avg_ops,
            "escalation_rate": esc_rate,
            "savings_vs_fixed_high_percent": savings,
            "by_case": by_case,
            "by_snr": by_snr,
        }

        print(f"{m:<18} | {m_metrics['f1']:.4f} | {m_metrics['recall']:.4f} | {m_metrics['false_alarm_rate']:.4f} | {fmae:8.2f} | {esc_rate*100:7.2f}% | {avg_ops:8.1f} | {savings:6.2f}%")

    # Run group-level bootstrap for 95% CIs
    print("\nComputing group-level bootstrap 95% confidence intervals (1,000 iterations)...")
    boot_ci = run_group_bootstrap(test_records, predictions, n_bootstraps=1000, seed=20261013)
    evaluation_summary["bootstrap_95ci"] = boot_ci

    print("\n--- 95% Bootstrap Confidence Intervals (Group-Level Resampling) ---")
    for m in ["fixed_low", "fixed_high", "adaptive_v1.0", "candidate_v1.1"]:
        ci = boot_ci[m]
        print(f"{m:<18}: F1=[{ci['f1'][0]:.4f}, {ci['f1'][1]:.4f}], Rec=[{ci['recall'][0]:.4f}, {ci['recall'][1]:.4f}], FAR=[{ci['false_alarm_rate'][0]:.4f}, {ci['false_alarm_rate'][1]:.4f}], Ops=[{ci['avg_ops'][0]:.1f}, {ci['avg_ops'][1]:.1f}]")

    with summary_path.open("x", encoding="utf-8") as f:
        json.dump(evaluation_summary, f, indent=2)

    return evaluation_summary


def main():
    parser = argparse.ArgumentParser(description="Phase 5 Final Held-Out Evaluation")
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args()
    run_phase5_evaluation(out_dir=args.out)


if __name__ == "__main__":
    main()
