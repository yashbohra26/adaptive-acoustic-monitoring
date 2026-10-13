"""Phase 3 pilot runner: Evaluate AdaptiveSelector on Development Data Only.

This script:
1. Loads/builds the 80-group pilot dataset (60 development groups: 30 fit, 30 calibration).
2. Fits FixedLowDetector and FixedHighDetector on normal model-fit windows.
3. Calibrates fixed thresholds on normal calibration windows.
4. Calibrates AdaptiveSelector boundaries (low_score_p90, low_score_p99, zcf_p01, zcf_p99)
   EXCLUSIVELY on normal calibration windows.
5. Evaluates adaptive decision flow and work metrics on DEVELOPMENT windows only.
6. Does NOT evaluate or inspect held-out test data in Phase 3.
7. Computes and saves escalation rate diagnostics overall, by case, and by SNR.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from collections import defaultdict
import numpy as np

from config import DEFAULT_CONFIG
from generate_signals import build_dataset, write_metadata
from baselines import FixedLowDetector, FixedHighDetector
from adaptive_selector import AdaptiveSelector


def run_phase3_pilot(
    out_dir: Path = Path("results/phase3_pilot"),
    groups: int = 80,
    target_far: float = 0.05,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating/Loading dataset with {groups} source groups...")
    records, signals = build_dataset(
        groups=groups,
        config=DEFAULT_CONFIG,
        dev_calibration_fraction=0.5,
    )

    dev_records = [r for r in records if r["split"] == "development"]
    test_records = [r for r in records if r["split"] == "test"]
    fit_records = [r for r in dev_records if r["dev_subset"] == "fit"]
    calib_records = [r for r in dev_records if r["dev_subset"] == "calibration"]

    print(f"Development windows: {len(dev_records)} across {len({r['group_id'] for r in dev_records})} groups")
    print(f"Held-out test windows: {len(test_records)} across {len({r['group_id'] for r in test_records})} groups (UNTOUCHED)")

    # Normal signals for fitting and calibration
    fit_normals = [signals[r["signal_key"]] for r in fit_records if r["case"] == "normal"]
    calib_normals = [signals[r["signal_key"]] for r in calib_records if r["case"] == "normal"]

    # 1. Fit & calibrate fixed baselines
    low = FixedLowDetector(sample_rate_hz=DEFAULT_CONFIG.sample_rate_hz)
    low.fit(fit_normals)
    low.calibrate(calib_normals, target_far=target_far)

    high = FixedHighDetector(sample_rate_hz=DEFAULT_CONFIG.sample_rate_hz)
    high.fit(fit_normals)
    high.calibrate(calib_normals, target_far=target_far)

    # 2. Calibrate Adaptive Selector boundaries
    selector = AdaptiveSelector(low, high, sample_rate_hz=DEFAULT_CONFIG.sample_rate_hz)
    boundaries = selector.calibrate_selector(calib_normals)

    print("\n--- Adaptive Selector Boundaries (from Normal Calibration Data) ---")
    print(f"Low score 90th percentile: {boundaries['low_score_p90']:.4f}")
    print(f"Low score 99th percentile: {boundaries['low_score_p99']:.4f}")
    print(f"Zero-crossing frequency 1st percentile: {boundaries['zcf_p01']:.2f} Hz")
    print(f"Zero-crossing frequency 99th percentile: {boundaries['zcf_p99']:.2f} Hz")

    # 3. Evaluate on Development Data Only
    dev_results = []
    fft_count_overall = 0
    total_ops_overall = 0
    total_time_overall = 0.0

    by_case = defaultdict(lambda: {"total": 0, "fft_used": 0, "stage1": 0, "stage2": 0})
    by_snr = defaultdict(lambda: {"total": 0, "fft_used": 0, "stage1": 0, "stage2": 0})
    by_case_snr = defaultdict(lambda: {"total": 0, "fft_used": 0, "stage1": 0, "stage2": 0})

    for r in dev_records:
        x = signals[r["signal_key"]]
        res = selector.predict(x)

        dev_results.append({
            "group_id": r["group_id"],
            "dev_subset": r["dev_subset"],
            "case": r["case"],
            "anomaly_ground_truth": r["anomaly"],
            "snr_db": r["snr_db"],
            "target_frequency_hz": r["target_frequency_hz"],
            "predicted_label": res.predicted_label,
            "final_anomaly_score": res.final_anomaly_score,
            "frequency_estimate_hz": res.frequency_estimate_hz,
            "stage_reached": res.stage_reached,
            "fft_used": int(res.fft_used),
            "escalation_reason": res.escalation_reason,
            "low_stage_score": res.low_stage_score,
            "high_stage_score": "" if res.high_stage_score is None else res.high_stage_score,
            "op_count_proxy": res.op_count_proxy,
            "execution_time_s": f"{res.execution_time_s:.6f}",
        })

        case = r["case"]
        snr = r["snr_db"]
        fft_used = res.fft_used

        if fft_used:
            fft_count_overall += 1
        total_ops_overall += res.op_count_proxy
        total_time_overall += res.execution_time_s

        for stat_dict in [by_case[case], by_snr[snr], by_case_snr[(case, snr)]]:
            stat_dict["total"] += 1
            if fft_used:
                stat_dict["fft_used"] += 1
                stat_dict["stage2"] += 1
            else:
                stat_dict["stage1"] += 1

    overall_escalation_rate = fft_count_overall / len(dev_records)
    avg_ops_adaptive = total_ops_overall / len(dev_records)

    # For comparison, calculate fixed baseline operation counts for N=1024
    n_samples = DEFAULT_CONFIG.samples_per_window
    ops_fixed_low_only = 10 * n_samples + 25
    n_log2_n = int(n_samples * np.log2(n_samples))
    ops_fixed_high_only = int(2.5 * n_log2_n + 3.5 * n_samples + 66)

    print(f"\n--- Development Escalation Summary ({len(dev_records)} windows) ---")
    print(f"Overall Escalation Rate: {overall_escalation_rate * 100:.2f}% ({fft_count_overall}/{len(dev_records)})")
    print(f"Average Operation Count Proxy: {avg_ops_adaptive:.0f}")
    print(f"  Fixed-Low Baseline Ops: {ops_fixed_low_only} per window")
    print(f"  Fixed-High Baseline Ops: {ops_fixed_high_only} per window")

    print("\nEscalation Rate by Case:")
    case_summary = {}
    for case, stats in sorted(by_case.items()):
        rate = stats["fft_used"] / stats["total"]
        case_summary[case] = {
            "total_windows": stats["total"],
            "escalated_count": stats["fft_used"],
            "escalation_rate": rate,
        }
        print(f"  {case:<16}: {rate * 100:6.2f}% ({stats['fft_used']}/{stats['total']})")

    print("\nEscalation Rate by SNR:")
    snr_summary = {}
    for snr, stats in sorted(by_snr.items(), reverse=True):
        rate = stats["fft_used"] / stats["total"]
        snr_summary[str(snr)] = {
            "total_windows": stats["total"],
            "escalated_count": stats["fft_used"],
            "escalation_rate": rate,
        }
        print(f"  SNR {snr:4.1f} dB : {rate * 100:6.2f}% ({stats['fft_used']}/{stats['total']})")

    # Save summary diagnostics
    diagnostics = {
        "dataset_summary": {
            "total_groups": groups,
            "dev_groups": len({r['group_id'] for r in dev_records}),
            "dev_windows": len(dev_records),
            "test_groups": len({r['group_id'] for r in test_records}),
            "test_windows": len(test_records),
            "test_status": "UNTOUCHED_AND_UNINSPECTED",
        },
        "selector_boundaries": boundaries,
        "overall_development_work_accounting": {
            "total_windows": len(dev_records),
            "fft_escalations": fft_count_overall,
            "overall_escalation_rate": overall_escalation_rate,
            "avg_op_count_proxy": avg_ops_adaptive,
            "fixed_low_op_count_proxy": ops_fixed_low_only,
            "fixed_high_op_count_proxy": ops_fixed_high_only,
            "avg_runtime_ms": (total_time_overall / len(dev_records)) * 1000.0,
        },
        "escalation_by_case": case_summary,
        "escalation_by_snr": snr_summary,
    }

    with (out_dir / "adaptive_diagnostics.json").open("w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2)

    with (out_dir / "development_adaptive_predictions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(dev_results[0].keys()))
        writer.writeheader()
        writer.writerows(dev_results)

    print(f"\nPhase 3 pilot outputs saved to {out_dir}:")
    print("  - adaptive_diagnostics.json")
    print("  - development_adaptive_predictions.csv")


def main():
    parser = argparse.ArgumentParser(description="Phase 3 adaptive selector runner")
    parser.add_argument("--out", type=Path, default=Path("results/phase3_pilot"))
    parser.add_argument("--groups", type=int, default=80)
    parser.add_argument("--far", type=float, default=0.05)
    args = parser.parse_args()

    run_phase3_pilot(out_dir=args.out, groups=args.groups, target_far=args.far)


if __name__ == "__main__":
    main()
