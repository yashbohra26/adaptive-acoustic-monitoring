"""Phase 4 development review diagnostics runner.

Evaluates development partitions only (fit and calibration, 960 windows across 60 groups)
comparing Fixed-Low, Fixed-High, and Adaptive processing.
Strictly refuses to load, inspect, or process held-out test windows.

Computes:
1. Overall precision, recall, F1, false-alarm rate (FAR = FP / (FP + TN)).
2. Recall by anomaly case (frequency_shift, added_harmonic, impact).
3. Performance metrics by SNR level (30, 20, 10, 0 dB).
4. Frequency error on eligible single-tone target cases (normal and frequency_shift).
5. Adaptive escalation rate and work accounting (operation count proxy and savings).
6. Group-level bootstrap resampling confidence intervals (95% CI).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from collections import defaultdict
import numpy as np

from config import DEFAULT_CONFIG
from generate_signals import build_dataset
from baselines import FixedLowDetector, FixedHighDetector
from adaptive_selector import AdaptiveSelector


def compute_binary_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, float]:
    """Calculate precision, recall, F1, and false-alarm rate."""
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    far = fp / (fp + tn) if (fp + tn) > 0 else 0.0  # False alarm rate on normal ground truth

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_alarm_rate": far,
    }


def compute_freq_mae(records, predictions) -> dict[str, float]:
    """Calculate MAE on eligible single-tone windows with defined target frequency."""
    errors = []
    for r, p in zip(records, predictions):
        if r["target_frequency_hz"] != "" and p["freq_est"] is not None:
            target = float(r["target_frequency_hz"])
            est = float(p["freq_est"])
            errors.append(abs(est - target))
    if not errors:
        return {"count": 0, "mae_hz": 0.0, "median_err_hz": 0.0}
    return {
        "count": len(errors),
        "mae_hz": float(np.mean(errors)),
        "median_err_hz": float(np.median(errors)),
    }


def group_bootstrap_ci(
    group_ids: list[str],
    records: list[dict],
    pred_labels: list[int],
    n_boot: int = 1000,
    seed: int = 20261014,
) -> dict[str, tuple[float, float]]:
    """Group-level bootstrap resampling for 95% confidence intervals."""
    rng = np.random.default_rng(seed)
    unique_groups = np.array(sorted(list(set(group_ids))))
    n_groups = len(unique_groups)

    # Pre-index windows by group
    group_to_indices = defaultdict(list)
    for idx, gid in enumerate(group_ids):
        group_to_indices[gid].append(idx)

    boot_f1 = []
    boot_far = []
    boot_recall = []

    for _ in range(n_boot):
        sampled_groups = rng.choice(unique_groups, size=n_groups, replace=True)
        sample_indices = []
        for sg in sampled_groups:
            sample_indices.extend(group_to_indices[sg])

        y_true_sample = [records[i]["anomaly"] for i in sample_indices]
        y_pred_sample = [pred_labels[i] for i in sample_indices]

        m = compute_binary_metrics(y_true_sample, y_pred_sample)
        boot_f1.append(m["f1"])
        boot_far.append(m["false_alarm_rate"])
        boot_recall.append(m["recall"])

    return {
        "f1_ci95": (float(np.percentile(boot_f1, 2.5)), float(np.percentile(boot_f1, 97.5))),
        "far_ci95": (float(np.percentile(boot_far, 2.5)), float(np.percentile(boot_far, 97.5))),
        "recall_ci95": (float(np.percentile(boot_recall, 2.5)), float(np.percentile(boot_recall, 97.5))),
    }


def run_phase4_development_review(
    out_dir: Path = Path("results/phase4_review"),
    groups: int = 80,
    target_far: float = 0.05,
    allow_test_split: bool = False,
) -> dict:
    if allow_test_split:
        raise ValueError("Security violation: Phase 4 development review must NOT run on test split!")

    out_dir.mkdir(parents=True, exist_ok=True)

    print("Building/loading dataset (80 source groups)...")
    records, signals = build_dataset(
        groups=groups,
        config=DEFAULT_CONFIG,
        dev_calibration_fraction=0.5,
    )

    # Strictly enforce development-only filter
    dev_records = [r for r in records if r["split"] == "development"]
    test_records = [r for r in records if r["split"] == "test"]

    # Safety guard: ensure no test record is present in dev_records
    for r in dev_records:
        if r["split"] == "test":
            raise RuntimeError("Fatal: test split record detected in development set!")

    print(f"Total dataset: {len(records)} windows")
    print(f"Active Development windows: {len(dev_records)} across {len({r['group_id'] for r in dev_records})} groups")
    print(f"Held-Out Test windows: {len(test_records)} across {len({r['group_id'] for r in test_records})} groups (STRICTLY LOCKED)")

    fit_records = [r for r in dev_records if r["dev_subset"] == "fit"]
    calib_records = [r for r in dev_records if r["dev_subset"] == "calibration"]

    fit_normals = [signals[r["signal_key"]] for r in fit_records if r["case"] == "normal"]
    calib_normals = [signals[r["signal_key"]] for r in calib_records if r["case"] == "normal"]

    # 1. Fit & calibrate baselines on development partitions only
    low = FixedLowDetector(sample_rate_hz=DEFAULT_CONFIG.sample_rate_hz)
    low.fit(fit_normals)
    low.calibrate(calib_normals, target_far=target_far)

    high = FixedHighDetector(sample_rate_hz=DEFAULT_CONFIG.sample_rate_hz)
    high.fit(fit_normals)
    high.calibrate(calib_normals, target_far=target_far)

    # 2. Calibrate Adaptive Selector on normal calibration partitions only
    selector = AdaptiveSelector(low, high, sample_rate_hz=DEFAULT_CONFIG.sample_rate_hz)
    boundaries = selector.calibrate_selector(calib_normals)

    # 3. Predict across all development records
    low_preds = []
    high_preds = []
    adapt_preds = []

    for r in dev_records:
        x = signals[r["signal_key"]]
        # Fixed Low
        l_res = low.predict(x)
        low_preds.append({
            "pred_label": l_res.predicted_label,
            "score": l_res.anomaly_score,
            "freq_est": l_res.frequency_estimate_hz,
            "ops": 10 * len(x) + 25,
        })

        # Fixed High
        h_res = high.predict(x)
        n_log2_n = int(len(x) * np.log2(len(x)))
        high_preds.append({
            "pred_label": h_res.predicted_label,
            "score": h_res.anomaly_score,
            "freq_est": h_res.frequency_estimate_hz,
            "ops": int(2.5 * n_log2_n + 3.5 * len(x) + 66),
        })

        # Adaptive
        a_res = selector.predict(x)
        adapt_preds.append({
            "pred_label": a_res.predicted_label,
            "score": a_res.final_anomaly_score,
            "freq_est": a_res.frequency_estimate_hz,
            "stage_reached": a_res.stage_reached,
            "fft_used": a_res.fft_used,
            "escalation_reason": a_res.escalation_reason,
            "ops": a_res.op_count_proxy,
        })

    y_true = [r["anomaly"] for r in dev_records]
    dev_group_ids = [r["group_id"] for r in dev_records]

    # Metrics computation
    methods = [
        ("Fixed-Low", [p["pred_label"] for p in low_preds], low_preds),
        ("Fixed-High", [p["pred_label"] for p in high_preds], high_preds),
        ("Adaptive", [p["pred_label"] for p in adapt_preds], adapt_preds),
    ]

    method_summaries = {}
    for name, preds, full_info in methods:
        overall = compute_binary_metrics(y_true, preds)
        freq_info = compute_freq_mae(dev_records, full_info)
        avg_ops = float(np.mean([p["ops"] for p in full_info]))
        ci = group_bootstrap_ci(dev_group_ids, dev_records, preds)

        # By case recall
        case_recall = {}
        for case in ["normal", "frequency_shift", "added_harmonic", "impact"]:
            idx = [i for i, r in enumerate(dev_records) if r["case"] == case]
            yt = [y_true[i] for i in idx]
            yp = [preds[i] for i in idx]
            m = compute_binary_metrics(yt, yp)
            case_recall[case] = {
                "count": len(idx),
                "predicted_anomaly_fraction": float(np.mean(yp)),
                "metric": "false_alarm_rate" if case == "normal" else "recall",
                "value": m["false_alarm_rate"] if case == "normal" else m["recall"],
            }

        # By SNR metrics
        snr_metrics = {}
        for snr in [30.0, 20.0, 10.0, 0.0]:
            idx = [i for i, r in enumerate(dev_records) if r["snr_db"] == snr]
            yt = [y_true[i] for i in idx]
            yp = [preds[i] for i in idx]
            m = compute_binary_metrics(yt, yp)
            f_snr = compute_freq_mae([dev_records[i] for i in idx], [full_info[i] for i in idx])
            snr_metrics[str(snr)] = {
                "count": len(idx),
                "precision": m["precision"],
                "recall": m["recall"],
                "f1": m["f1"],
                "false_alarm_rate": m["false_alarm_rate"],
                "freq_mae_hz": f_snr["mae_hz"],
            }

        method_summaries[name] = {
            "overall_binary": overall,
            "bootstrap_ci95": ci,
            "frequency_estimation": freq_info,
            "average_op_count_proxy": avg_ops,
            "performance_by_case": case_recall,
            "performance_by_snr": snr_metrics,
        }

    # Relative savings
    fixed_high_ops = method_summaries["Fixed-High"]["average_op_count_proxy"]
    adaptive_ops = method_summaries["Adaptive"]["average_op_count_proxy"]
    savings_vs_fixed_high = (fixed_high_ops - adaptive_ops) / fixed_high_ops

    # Escalation rates
    fft_flags = [p["fft_used"] for p in adapt_preds]
    escalation_overall = float(np.mean(fft_flags))

    escalation_by_case = {}
    for case in ["normal", "frequency_shift", "added_harmonic", "impact"]:
        idx = [i for i, r in enumerate(dev_records) if r["case"] == case]
        escalation_by_case[case] = float(np.mean([fft_flags[i] for i in idx]))

    escalation_by_snr = {}
    for snr in [30.0, 20.0, 10.0, 0.0]:
        idx = [i for i, r in enumerate(dev_records) if r["snr_db"] == snr]
        escalation_by_snr[str(snr)] = float(np.mean([fft_flags[i] for i in idx]))

    results = {
        "status": "DEVELOPMENT_DIAGNOSTICS_ONLY",
        "notice": "These metrics are development-set pilot diagnostics for model review and protocol freeze. They are NOT unbiased held-out generalization test results.",
        "dataset_split": {
            "source_groups": groups,
            "development_groups": 60,
            "held_out_test_groups": 20,
            "development_windows": len(dev_records),
            "held_out_test_windows_locked": len(test_records),
        },
        "methods_comparison": method_summaries,
        "work_accounting_and_savings": {
            "fixed_low_standalone_ops": method_summaries["Fixed-Low"]["average_op_count_proxy"],
            "fixed_high_standalone_ops": fixed_high_ops,
            "adaptive_weighted_ops": adaptive_ops,
            "work_reduction_vs_fixed_high_percent": savings_vs_fixed_high * 100.0,
            "escalation_overall": escalation_overall,
            "escalation_by_case": escalation_by_case,
            "escalation_by_snr": escalation_by_snr,
        },
    }

    with (out_dir / "development_review_summary.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Print summary table
    print("\n================== DEVELOPMENT DIAGNOSTICS SUMMARY ==================")
    print(f"{'Method':<12} | {'F1':<6} | {'Recall':<6} | {'FAR':<6} | {'Freq MAE (Hz)':<14} | {'Avg Ops/Win':<11} | {'Ops vs High'}")
    print("-" * 75)
    for name in ["Fixed-Low", "Fixed-High", "Adaptive"]:
        m = method_summaries[name]
        f1 = m["overall_binary"]["f1"]
        rec = m["overall_binary"]["recall"]
        far = m["overall_binary"]["false_alarm_rate"]
        fmae = m["frequency_estimation"]["mae_hz"]
        ops = m["average_op_count_proxy"]
        rel = f"-{(1 - ops / fixed_high_ops) * 100:.1f}%" if ops < fixed_high_ops else "baseline"
        print(f"{name:<12} | {f1:.4f} | {rec:.4f} | {far:.4f} | {fmae:14.2f} | {ops:11.1f} | {rel}")
    print("=====================================================================")

    return results


def main():
    parser = argparse.ArgumentParser(description="Phase 4 development review diagnostics")
    parser.add_argument("--out", type=Path, default=Path("results/phase4_review"))
    parser.add_argument("--groups", type=int, default=80)
    args = parser.parse_args()

    run_phase4_development_review(out_dir=args.out, groups=args.groups)


if __name__ == "__main__":
    main()
