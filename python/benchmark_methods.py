"""Fair, reproducible benchmarking harness across Fixed-Low, Fixed-High, and Adaptive detectors.

Measures all detectors under strictly identical conditions:
1. Identical input windows from the frozen 20 held-out test groups (320 windows).
2. Uniform timing methodology: 5 warmup iterations followed by 25 timing repetitions
   per window using time.perf_counter_ns().
3. Evaluates both detection quality (Accuracy, Precision, Recall, F1, FAR, Confusion counts)
   and computational work (Operation-count proxy and desktop runtime).
4. Tracks adaptive escalation pathways and decision frequencies.
5. Explicit disclaimers: Desktop Python runtime on host CPU does NOT represent
   microcontroller (ESP32) execution time or physical energy consumption.
"""
from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Sequence
import numpy as np
import pandas as pd

from baselines import FixedLowDetector, FixedHighDetector
from candidate_selector import CandidateAdaptiveSelector
from adaptive_selector import AdaptiveSelector
from config import DEFAULT_CONFIG
from generate_signals import build_dataset


def compute_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> dict[str, float | int]:
    """Compute binary classification metrics and confusion matrix counts."""
    yt = np.asarray(y_true, dtype=int)
    yp = np.asarray(y_pred, dtype=int)

    tp = int(np.sum((yt == 1) & (yp == 1)))
    fp = int(np.sum((yt == 0) & (yp == 1)))
    tn = int(np.sum((yt == 0) & (yp == 0)))
    fn = int(np.sum((yt == 1) & (yp == 0)))

    total = len(yt)
    accuracy = float((tp + tn) / total) if total > 0 else 0.0
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    far = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_alarm_rate": far,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def compute_freq_mae(records: Sequence[dict], freqs: Sequence[float | None]) -> float | None:
    """Compute frequency MAE strictly on eligible single-tone windows."""
    diffs = []
    for r, f in zip(records, freqs):
        if r["target_frequency_hz"] != "" and f is not None:
            target = float(r["target_frequency_hz"])
            diffs.append(abs(float(f) - target))
    return float(np.mean(diffs)) if diffs else None


def benchmark_method_window(detector, x: np.ndarray, n_warmup: int = 5, n_reps: int = 25) -> tuple[any, float, float]:
    """Execute prediction with warmup and repeated nano-timing."""
    # Warmup
    for _ in range(n_warmup):
        detector.predict(x)

    # Nano-timing
    durations_ns = []
    result = None
    for _ in range(n_reps):
        t0 = time.perf_counter_ns()
        res = detector.predict(x)
        t1 = time.perf_counter_ns()
        durations_ns.append(t1 - t0)
        result = res

    mean_ms = float(np.mean(durations_ns) / 1e6)
    std_ms = float(np.std(durations_ns) / 1e6)
    return result, mean_ms, std_ms


def run_fair_benchmark(out_dir: Path = Path("results")) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 80)
    print("FAIR THREE-METHOD BENCHMARK HARNESS")
    print("=" * 80)

    print("Building full 80-group dataset...")
    records, signals = build_dataset(groups=80, config=DEFAULT_CONFIG, dev_calibration_fraction=0.5)

    dev_records = [r for r in records if r["split"] == "development"]
    test_records = [r for r in records if r["split"] == "test"]
    assert len(dev_records) == 960
    assert len(test_records) == 320

    # Fit & calibrate models strictly on dev
    fit_sigs = [signals[r["signal_key"]] for r in dev_records if r["dev_subset"] == "fit" and r["case"] == "normal"]
    calib_sigs = [signals[r["signal_key"]] for r in dev_records if r["dev_subset"] == "calibration" and r["case"] == "normal"]
    calib_snrs = [r["snr_db"] for r in dev_records if r["dev_subset"] == "calibration" and r["case"] == "normal"]

    print("Fitting and calibrating detectors on development partition...")
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

    hist_det = AdaptiveSelector(low_det, high_det, sample_rate_hz=DEFAULT_CONFIG.sample_rate_hz)
    hist_det.calibrate_selector(calib_sigs)

    detectors = {
        "fixed_low": low_det,
        "fixed_high": high_det,
        "candidate_v1.1": cand_det,
        "historical_v1.0": hist_det,
    }

    print("Executing benchmark across all 320 test windows (warmup=5, reps=25)...")
    window_rows = []
    method_data = {m: {"preds": [], "scores": [], "freqs": [], "stages": [], "ffts": [], "reasons": [], "ops": [], "runtimes_ms": []} for m in detectors}

    y_true = [r["anomaly"] for r in test_records]

    for idx, r in enumerate(test_records):
        key = r["signal_key"]
        x = signals[key]
        row = {
            "window_idx": idx,
            "group_id": r["group_id"],
            "case": r["case"],
            "snr_db": r["snr_db"],
            "anomaly_true": r["anomaly"],
            "target_freq_hz": r["target_frequency_hz"],
        }

        for m_name, det in detectors.items():
            res, mean_ms, std_ms = benchmark_method_window(det, x, n_warmup=5, n_reps=25)

            if m_name == "fixed_low":
                pred = res.predicted_label
                score = res.anomaly_score
                freq = res.frequency_estimate_hz
                stage = "stage1_low"
                fft_used = False
                reason = "none"
                ops = 10265
            elif m_name == "fixed_high":
                pred = res.predicted_label
                score = res.anomaly_score
                freq = res.frequency_estimate_hz
                stage = "stage2_high"
                fft_used = True
                reason = "none"
                ops = 29250
            else:
                pred = res.predicted_label
                score = res.final_anomaly_score
                freq = res.frequency_estimate_hz
                stage = res.stage_reached
                fft_used = res.fft_used
                reason = res.escalation_reason
                ops = res.op_count_proxy

            method_data[m_name]["preds"].append(pred)
            method_data[m_name]["scores"].append(score)
            method_data[m_name]["freqs"].append(freq)
            method_data[m_name]["stages"].append(stage)
            method_data[m_name]["ffts"].append(fft_used)
            method_data[m_name]["reasons"].append(reason)
            method_data[m_name]["ops"].append(ops)
            method_data[m_name]["runtimes_ms"].append(mean_ms)

            row[f"{m_name}_pred"] = pred
            row[f"{m_name}_score"] = score
            row[f"{m_name}_freq_hz"] = freq
            row[f"{m_name}_stage"] = stage
            row[f"{m_name}_fft"] = fft_used
            row[f"{m_name}_reason"] = reason
            row[f"{m_name}_ops"] = ops
            row[f"{m_name}_runtime_ms"] = mean_ms

        window_rows.append(row)

    df_windows = pd.DataFrame(window_rows)
    df_windows.to_csv(out_dir / "benchmark_fair_windows.csv", index=False)
    print(f"Saved window predictions to {out_dir / 'benchmark_fair_windows.csv'}")

    # Aggregate summaries
    benchmark_summary = {
        "status": "FAIR_BENCHMARK_COMPLETE",
        "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "test_windows": len(test_records),
        "test_groups": len(set(r["group_id"] for r in test_records)),
        "timing_parameters": {
            "warmup_iterations": 5,
            "timed_repetitions": 25,
            "timer": "time.perf_counter_ns",
            "unit": "milliseconds (ms)",
            "disclaimer": "Measured on host CPU under Python 3.13; does NOT represent ESP32 runtime or physical energy.",
        },
        "methods": {},
    }

    print("\n" + "=" * 80)
    print(f"{'Method':<18} | {'F1':<6} | {'Recall':<6} | {'FAR':<6} | {'Freq MAE':<9} | {'Avg Ops':<9} | {'Savings':<8} | {'Runtime (ms)':<12}")
    print("-" * 80)

    fixed_high_ops = 29250.0

    for m_name in ["fixed_low", "fixed_high", "candidate_v1.1", "historical_v1.0"]:
        md = method_data[m_name]
        metrics = compute_metrics(y_true, md["preds"])
        fmae = compute_freq_mae(test_records, md["freqs"])
        avg_ops = float(np.mean(md["ops"]))
        avg_rt = float(np.mean(md["runtimes_ms"]))
        std_rt = float(np.std(md["runtimes_ms"]))
        esc_rate = float(np.mean(md["ffts"]))
        savings = float((fixed_high_ops - avg_ops) / fixed_high_ops * 100.0)

        # Escalation breakdown by reason
        reasons_count = {}
        for r in md["reasons"]:
            reasons_count[r] = reasons_count.get(r, 0) + 1

        # Breakdown by case
        by_case = {}
        for c in ["normal", "frequency_shift", "added_harmonic", "impact"]:
            c_idx = [i for i, r in enumerate(test_records) if r["case"] == c]
            c_yt = [y_true[i] for i in c_idx]
            c_yp = [md["preds"][i] for i in c_idx]
            c_metrics = compute_metrics(c_yt, c_yp)
            c_ops = float(np.mean([md["ops"][i] for i in c_idx]))
            c_rt = float(np.mean([md["runtimes_ms"][i] for i in c_idx]))
            c_esc = float(np.mean([md["ffts"][i] for i in c_idx]))

            by_case[c] = {
                "count": len(c_idx),
                "metrics": c_metrics,
                "avg_ops": c_ops,
                "avg_runtime_ms": c_rt,
                "escalation_rate": c_esc,
            }

        # Breakdown by SNR
        by_snr = {}
        for snr in [30.0, 20.0, 10.0, 0.0]:
            s_idx = [i for i, r in enumerate(test_records) if r["snr_db"] == snr]
            s_yt = [y_true[i] for i in s_idx]
            s_yp = [md["preds"][i] for i in s_idx]
            s_metrics = compute_metrics(s_yt, s_yp)
            s_ops = float(np.mean([md["ops"][i] for i in s_idx]))
            s_rt = float(np.mean([md["runtimes_ms"][i] for i in s_idx]))
            s_esc = float(np.mean([md["ffts"][i] for i in s_idx]))

            by_snr[str(snr)] = {
                "count": len(s_idx),
                "metrics": s_metrics,
                "avg_ops": s_ops,
                "avg_runtime_ms": s_rt,
                "escalation_rate": s_esc,
            }

        benchmark_summary["methods"][m_name] = {
            "metrics": metrics,
            "freq_mae_hz": fmae,
            "avg_op_count_proxy": avg_ops,
            "savings_vs_fixed_high_percent": savings,
            "escalation_rate": esc_rate,
            "escalation_reason_counts": reasons_count,
            "runtime_ms": {
                "mean": avg_rt,
                "std": std_rt,
            },
            "by_case": by_case,
            "by_snr": by_snr,
        }

        fmae_str = f"{fmae:.2f} Hz" if fmae is not None else "N/A"
        print(f"{m_name:<18} | {metrics['f1']:<6.4f} | {metrics['recall']:<6.4f} | {metrics['false_alarm_rate']:<6.4f} | {fmae_str:<9} | {avg_ops:<9.1f} | {savings:<+7.2f}% | {avg_rt:.4f} +- {std_rt:.4f}")

    print("=" * 80)

    summary_file = out_dir / "benchmark_fair_evaluation.json"
    with summary_file.open("w", encoding="utf-8") as f:
        json.dump(benchmark_summary, f, indent=2)
    print(f"Saved comprehensive benchmark summary to {summary_file}")

    return benchmark_summary


if __name__ == "__main__":
    run_fair_benchmark()
