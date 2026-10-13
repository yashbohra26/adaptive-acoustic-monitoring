"""Phase 2 pilot runner: Fit and calibrate fixed baselines on development data.

This script:
1. Generates 80 source groups (60 development, 20 held-out test).
2. Partitions development groups into model-fit (30 groups) and threshold-calibration (30 groups).
3. Fits Fixed-Low and Fixed-High baseline reference parameters on normal model-fit data only.
4. Calibrates decision thresholds on normal threshold-calibration data targeting 5% false-alarm rate.
5. Saves pilot datasets, reference parameters, calibration details, and development-only validation summaries.
6. Does NOT evaluate or inspect held-out test data in Phase 2.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import numpy as np

from config import DEFAULT_CONFIG
from generate_signals import build_dataset, write_metadata
from baselines import FixedLowDetector, FixedHighDetector


def run_phase2_pilot(
    out_dir: Path = Path("results/phase2_pilot"),
    groups: int = 80,
    target_far: float = 0.05,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating Phase 2 pilot dataset: {groups} source groups...")
    # 80 groups with default test_fraction=0.25 -> 20 test groups, 60 dev groups.
    # dev_calibration_fraction=0.5 -> 30 model-fit groups, 30 threshold-calibration groups.
    records, signals = build_dataset(
        groups=groups,
        config=DEFAULT_CONFIG,
        dev_calibration_fraction=0.5,
    )

    # Separate metadata records by split
    dev_records = [r for r in records if r["split"] == "development"]
    test_records = [r for r in records if r["split"] == "test"]
    fit_records = [r for r in dev_records if r["dev_subset"] == "fit"]
    calib_records = [r for r in dev_records if r["dev_subset"] == "calibration"]

    write_metadata(dev_records, out_dir / "development_metadata.csv")
    write_metadata(test_records, out_dir / "test_metadata.csv")
    write_metadata(fit_records, out_dir / "fit_metadata.csv")
    write_metadata(calib_records, out_dir / "calibration_metadata.csv")

    dev_groups = sorted(list({r["group_id"] for r in dev_records}))
    test_groups = sorted(list({r["group_id"] for r in test_records}))
    fit_groups = sorted(list({r["group_id"] for r in fit_records}))
    calib_groups = sorted(list({r["group_id"] for r in calib_records}))

    print(f"Total windows: {len(records)}")
    print(f"Development windows: {len(dev_records)} across {len(dev_groups)} groups")
    print(f"  - Model-fit: {len(fit_records)} windows ({len(fit_groups)} groups)")
    print(f"  - Calibration: {len(calib_records)} windows ({len(calib_groups)} groups)")
    print(f"Held-out test windows: {len(test_records)} across {len(test_groups)} groups (UNTOUCHED)")

    # Extract signals for normal fit and normal calibration
    fit_normal_signals = [
        signals[r["signal_key"]]
        for r in fit_records
        if r["case"] == "normal"
    ]
    calib_normal_signals = [
        signals[r["signal_key"]]
        for r in calib_records
        if r["case"] == "normal"
    ]

    print(f"\nNormal fit signals available: {len(fit_normal_signals)}")
    print(f"Normal calibration signals available: {len(calib_normal_signals)}")

    # 1. Fixed-Low Detector
    low_detector = FixedLowDetector(sample_rate_hz=DEFAULT_CONFIG.sample_rate_hz)
    low_detector.fit(fit_normal_signals)
    low_thresh = low_detector.calibrate(calib_normal_signals, target_far=target_far)

    # 2. Fixed-High Detector
    high_detector = FixedHighDetector(sample_rate_hz=DEFAULT_CONFIG.sample_rate_hz)
    high_detector.fit(fit_normal_signals)
    high_thresh = high_detector.calibrate(calib_normal_signals, target_far=target_far)

    print("\n--- Baseline Calibration Details ---")
    print(f"Target Calibration False-Alarm Rate: {target_far * 100:.1f}%")
    print(f"Fixed-Low provisional threshold: {low_thresh:.4f}")
    print(f"Fixed-High provisional threshold: {high_thresh:.4f}")

    # Evaluate empirically on the normal calibration set
    low_calib_scores = [low_detector.score_window(x) for x in calib_normal_signals]
    high_calib_scores = [high_detector.score_window(x) for x in calib_normal_signals]

    low_calib_false_alarms = sum(s > low_thresh for s in low_calib_scores)
    high_calib_false_alarms = sum(s > high_thresh for s in high_calib_scores)
    empirical_low_far = low_calib_false_alarms / len(low_calib_scores)
    empirical_high_far = high_calib_false_alarms / len(high_calib_scores)

    print(f"Fixed-Low empirical FAR on calibration set: {empirical_low_far:.3f} ({low_calib_false_alarms}/{len(low_calib_scores)})")
    print(f"Fixed-High empirical FAR on calibration set: {empirical_high_far:.3f} ({high_calib_false_alarms}/{len(high_calib_scores)})")

    # Save calibration & parameters artifact
    calib_info = {
        "dataset_summary": {
            "total_groups": groups,
            "dev_groups": len(dev_groups),
            "test_groups": len(test_groups),
            "dev_fit_groups": len(fit_groups),
            "dev_calib_groups": len(calib_groups),
            "total_windows": len(records),
            "fit_normal_windows": len(fit_normal_signals),
            "calib_normal_windows": len(calib_normal_signals),
        },
        "target_calibration_far": target_far,
        "note": "Provisional thresholds targeting calibration FAR. Held-out test performance is NOT evaluated in Phase 2.",
        "fixed_low": {
            "feature_names": ["log_rms", "log_crest_factor", "log_kurtosis"],
            "reference_median": low_detector.reference_median.tolist(),
            "reference_scale": low_detector.reference_scale.tolist(),
            "provisional_threshold": low_thresh,
            "calibration_set_empirical_far": empirical_low_far,
            "calibration_set_normal_count": len(calib_normal_signals),
        },
        "fixed_high": {
            "num_bands": 16,
            "freq_range_hz": [0.0, 1200.0],
            "reference_median": high_detector.reference_median.tolist(),
            "reference_scale": high_detector.reference_scale.tolist(),
            "provisional_threshold": high_thresh,
            "calibration_set_empirical_far": empirical_high_far,
            "calibration_set_normal_count": len(calib_normal_signals),
        },
    }

    with (out_dir / "pilot_calibration.json").open("w", encoding="utf-8") as f:
        json.dump(calib_info, f, indent=2)

    # Also compute development-set preview predictions (on all dev windows) for inspection
    dev_predictions = []
    for r in dev_records:
        x = signals[r["signal_key"]]
        low_res = low_detector.predict(x)
        high_res = high_detector.predict(x)

        dev_predictions.append({
            "group_id": r["group_id"],
            "dev_subset": r["dev_subset"],
            "case": r["case"],
            "anomaly_ground_truth": r["anomaly"],
            "snr_db": r["snr_db"],
            "target_frequency_hz": r["target_frequency_hz"],
            "low_score": low_res.anomaly_score,
            "low_pred": low_res.predicted_label,
            "low_freq_est": low_res.frequency_estimate_hz,
            "high_score": high_res.anomaly_score,
            "high_pred": high_res.predicted_label,
            "high_freq_est": high_res.frequency_estimate_hz,
        })

    with (out_dir / "development_predictions_preview.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(dev_predictions[0].keys()))
        writer.writeheader()
        writer.writerows(dev_predictions)

    print(f"\nPhase 2 pilot artifacts successfully saved to {out_dir}:")
    print("  - development_metadata.csv")
    print("  - test_metadata.csv (held out)")
    print("  - fit_metadata.csv")
    print("  - calibration_metadata.csv")
    print("  - pilot_calibration.json")
    print("  - development_predictions_preview.csv")


def main():
    parser = argparse.ArgumentParser(description="Phase 2 baseline calibration runner")
    parser.add_argument("--out", type=Path, default=Path("results/phase2_pilot"))
    parser.add_argument("--groups", type=int, default=80)
    parser.add_argument("--far", type=float, default=0.05)
    args = parser.parse_args()

    run_phase2_pilot(out_dir=args.out, groups=args.groups, target_far=args.far)


if __name__ == "__main__":
    main()
