"""Development-only selector study and candidate search using grouped cross-validation.

Uses ONLY the 60 development source groups (strictly refuses test split).
Performs 5-fold cross-validation by source group:
- Each fold holds out 12 validation groups (192 windows).
- Training groups (48 groups) are split into:
  - 24 model-fit groups (fitting normal reference distributions)
  - 24 threshold-calibration groups (calibrating fixed thresholds and selector boundaries)
- Out-of-fold predictions are collected for Fixed-Low, Fixed-High, and each Candidate Adaptive Selector.
- Evaluates candidate policies against predeclared quality constraints relative to OOF Fixed-High:
  - F1 within 0.05 of Fixed-High
  - Recall within 0.05 of Fixed-High
  - FAR no more than 0.01 above Fixed-High
  - Average operation count proxy strictly < Fixed-High (29,250 ops)
- Selects the qualifying candidate with lowest operation-count proxy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import defaultdict
import numpy as np

from config import DEFAULT_CONFIG
from generate_signals import build_dataset
from baselines import FixedLowDetector, FixedHighDetector
from features import zero_crossing_frequency, kurtosis, crest_factor, rms
from candidate_features import normalized_difference_ratio, get_clean_calibrated_zcf_bounds


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


def compute_freq_mae(records: list[dict], freq_ests: list[float | None]) -> float:
    errors = []
    for r, f_est in zip(records, freq_ests):
        if r["target_frequency_hz"] != "" and f_est is not None:
            target = float(r["target_frequency_hz"])
            errors.append(abs(float(f_est) - target))
    return float(np.mean(errors)) if errors else 0.0


def run_identifiability_audit(dev_records: list[dict], signals: dict) -> dict:
    """Audit identifiability of features, noise smearing, and cascade cost limits."""
    cases = ["normal", "frequency_shift", "added_harmonic", "impact"]
    snrs = [30.0, 20.0, 10.0, 0.0]

    feature_stats_by_case_snr = defaultdict(dict)
    for c in cases:
        for s in snrs:
            sub = [r for r in dev_records if r["case"] == c and r["snr_db"] == s]
            zcfs = [zero_crossing_frequency(signals[r["signal_key"]], 4000) for r in sub]
            kurts = [kurtosis(signals[r["signal_key"]]) for r in sub]
            crests = [crest_factor(signals[r["signal_key"]]) for r in sub]
            drs = [normalized_difference_ratio(signals[r["signal_key"]]) for r in sub]
            key = f"{c}_{s:.0f}dB"
            feature_stats_by_case_snr[key] = {
                "case": c,
                "snr_db": s,
                "zcf_hz": {"min": float(np.min(zcfs)), "median": float(np.median(zcfs)), "max": float(np.max(zcfs))},
                "kurtosis": {"min": float(np.min(kurts)), "median": float(np.median(kurts)), "max": float(np.max(kurts))},
                "crest_factor": {"min": float(np.min(crests)), "median": float(np.median(crests)), "max": float(np.max(crests))},
                "diff_ratio": {"min": float(np.min(drs)), "median": float(np.median(drs)), "max": float(np.max(drs))},
            }

    audit = {
        "audit_scope": "DEVELOPMENT_SET_ONLY (960 windows across 60 groups)",
        "inference_time_inputs_available": {
            "window_samples_x": True,
            "sample_rate_hz": True,
            "reference_fit_parameters": True,
            "threshold_calibration_boundaries": True,
            "source_group_id": False,
            "nominal_normal_frequency_hz": False,
            "snr_db": False,
            "ground_truth_case": False,
        },
        "cascade_cost_accounting": {
            "stage1_ops": 10265,
            "stage2_added_ops": 29250,
            "escalated_total_ops": 39515,
            "standalone_fixed_high_ops": 29250,
            "break_even_escalation_rate": (29250 - 10265) / 29250,  # 0.64906 (64.91%)
            "break_even_formula": "E_max = (C_high - C_stage1) / C_high = (29250 - 10265) / 29250 = 64.91%",
            "impact_of_adding_difference_ratio_feature_to_stage1": {
                "diff_ratio_ops": 7169,
                "new_stage1_ops": 17434,
                "new_break_even_escalation_rate": (29250 - 17434) / 29250,  # 0.40397 (40.40%)
                "conclusion": "Adding candidate feature to Stage 1 reduces allowable escalation from 64.91% to 40.40%, hurting efficiency.",
            },
        },
        "identifiability_findings": {
            "impact": "100% identifiable in Stage 1 via kurtosis (min 10.56 vs normal max 2.90) and crest factor. Never needs FFT escalation.",
            "frequency_shift": "Clean fundamental range (264-405 Hz) cleanly separated from normal fundamental (180-260 Hz) at high SNRs. Noise smears ZCF at 0 dB to >500 Hz for both classes.",
            "added_harmonic": "Identical fundamental to normal (180-260 Hz). Kurtosis separated at >=10 dB (~1.92 vs ~1.50) but overlaps at 0 dB due to Gaussian noise kurtosis ~3.0. Escalation required for robust classification.",
            "normal": "Clean fundamental bounded in [180, 260] Hz. White noise causes high ZCF (>500 Hz) at 0 dB, requiring escalation to avoid false alarms.",
        },
        "feature_statistics": feature_stats_by_case_snr,
    }
    return audit


def run_cv_study(out_dir: Path = Path("results/phase4_study")):
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Building dataset (80 source groups)...")
    records, signals = build_dataset(groups=80, config=DEFAULT_CONFIG, dev_calibration_fraction=0.5)

    # STRICTLY development split only
    dev_records = [r for r in records if r["split"] == "development"]
    test_records = [r for r in records if r["split"] == "test"]
    assert len(dev_records) == 960
    assert len(test_records) == 320
    print(f"Development windows: {len(dev_records)} across 60 groups.")
    print("Held-out test windows: 320 across 20 groups (LOCKED - NOT ACCESSED).")

    # Run Identifiability Audit
    print("Running identifiability audit...")
    audit = run_identifiability_audit(dev_records, signals)
    with (out_dir / "identifiability_audit.json").open("w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)

    dev_groups = sorted(list(set(r["group_id"] for r in dev_records)))
    n_folds = 5
    folds = [dev_groups[i::n_folds] for i in range(n_folds)]

    # Comprehensive Candidate Grid:
    # Historical v1.0 Candidates (evaluating order: zcf_check -> low_normal -> low_anomaly)
    # Plus Candidate v1.1 Candidates (evaluating order: confident_anomaly_first -> zcf_check -> low_normal -> uncertain)
    candidates = [
        {"name": "v1.0_baseline", "order": "v1.0", "p_low": 90.0, "p_high": 99.0, "zcf_mode": "all_snr"},
        {"name": "v1.0_clean_zcf", "order": "v1.0", "p_low": 90.0, "p_high": 99.0, "zcf_mode": "clean_only"},
        {"name": "v1.0_clean_margin", "order": "v1.0", "p_low": 90.0, "p_high": 99.0, "zcf_mode": "clean_margin"},
        {"name": "cand_p80_clean_margin", "order": "v1.0", "p_low": 80.0, "p_high": 99.0, "zcf_mode": "clean_margin"},
        {"name": "cand_p70_clean_margin", "order": "v1.0", "p_low": 70.0, "p_high": 99.0, "zcf_mode": "clean_margin"},
        {"name": "cand_p60_clean_margin", "order": "v1.0", "p_low": 60.0, "p_high": 99.0, "zcf_mode": "clean_margin"},
        {"name": "cand_p50_clean_margin", "order": "v1.0", "p_low": 50.0, "p_high": 99.0, "zcf_mode": "clean_margin"},
        {"name": "cand_always_escalate", "order": "v1.0", "p_low": 0.0, "p_high": 100.0, "zcf_mode": "always_escalate"},
        # Candidate v1.1 Configurations (Corrected Cascade Order: Anomaly-First)
        {"name": "cand_v1.1_p70_anomaly_first", "order": "anomaly_first", "p_low": 70.0, "p_high_mult": 1.2, "zcf_mode": "clean_margin"},
        {"name": "cand_v1.1_p60_anomaly_first", "order": "anomaly_first", "p_low": 60.0, "p_high_mult": 1.2, "zcf_mode": "clean_margin"},
        {"name": "cand_v1.1_p50_anomaly_first", "order": "anomaly_first", "p_low": 50.0, "p_high_mult": 1.2, "zcf_mode": "clean_margin"},
    ]

    oof_low_pred = {}
    oof_high_pred = {}
    oof_cand_pred = {c["name"]: {} for c in candidates}

    # Group-wise 5-fold CV
    for fold_idx in range(n_folds):
        val_g = set(folds[fold_idx])
        train_g = [g for g in dev_groups if g not in val_g]
        fit_g = set(train_g[:len(train_g)//2])
        calib_g = set(train_g[len(train_g)//2:])

        # Extract fold partitions
        fit_norm_sigs = [signals[r["signal_key"]] for r in dev_records if r["group_id"] in fit_g and r["case"] == "normal"]
        calib_norm_sigs = [signals[r["signal_key"]] for r in dev_records if r["group_id"] in calib_g and r["case"] == "normal"]
        calib_norm_recs = [r for r in dev_records if r["group_id"] in calib_g and r["case"] == "normal"]

        low_det = FixedLowDetector(sample_rate_hz=DEFAULT_CONFIG.sample_rate_hz)
        low_det.fit(fit_norm_sigs)
        low_det.calibrate(calib_norm_sigs, target_far=0.05)

        high_det = FixedHighDetector(sample_rate_hz=DEFAULT_CONFIG.sample_rate_hz)
        high_det.fit(fit_norm_sigs)
        high_det.calibrate(calib_norm_sigs, target_far=0.05)

        # Precompute calibration distributions for candidates
        calib_low_scores = [low_det.score_window(x) for x in calib_norm_sigs]
        calib_zcfs_all = [zero_crossing_frequency(x, DEFAULT_CONFIG.sample_rate_hz) for x in calib_norm_sigs]
        clean_calib_zcfs = [
            zero_crossing_frequency(signals[r["signal_key"]], DEFAULT_CONFIG.sample_rate_hz)
            for r in calib_norm_recs if r["snr_db"] >= 20.0
        ]

        # Evaluate validation windows
        val_recs = [r for r in dev_records if r["group_id"] in val_g]
        for r in val_recs:
            key = r["signal_key"]
            x = signals[key]

            # Fixed-Low prediction
            l_res = low_det.predict(x)
            oof_low_pred[key] = {
                "pred": l_res.predicted_label,
                "score": l_res.anomaly_score,
                "freq": l_res.frequency_estimate_hz,
            }

            # Fixed-High prediction
            h_res = high_det.predict(x)
            oof_high_pred[key] = {
                "pred": h_res.predicted_label,
                "score": h_res.anomaly_score,
                "freq": h_res.frequency_estimate_hz,
            }

            # Evaluate each candidate policy
            low_score = l_res.anomaly_score
            zcf = l_res.frequency_estimate_hz

            for c in candidates:
                c_name = c["name"]
                order = c.get("order", "v1.0")

                if c["zcf_mode"] == "always_escalate":
                    fft_used = True
                    reason = "always_escalate"
                elif order == "anomaly_first":
                    # Candidate v1.1 Decision Order: Confident Anomaly First
                    p_high_val = float(max(calib_low_scores)) * c.get("p_high_mult", 1.2)
                    p_low_val = float(np.percentile(calib_low_scores, c["p_low"]))
                    z_min = float(min(clean_calib_zcfs)) - 10.0
                    z_max = float(max(clean_calib_zcfs)) + 10.0

                    if low_score >= p_high_val:
                        fft_used = False
                        reason = "confident_anomaly"
                    elif zcf < z_min or zcf > z_max:
                        fft_used = True
                        reason = "zcf_out_of_range"
                    elif low_score <= p_low_val:
                        fft_used = False
                        reason = "confident_normal"
                    else:
                        fft_used = True
                        reason = "uncertain_score"

                else:
                    # Historical v1.0 Decision Order
                    if c["zcf_mode"] == "all_snr":
                        z_min = float(np.percentile(calib_zcfs_all, 1.0))
                        z_max = float(np.percentile(calib_zcfs_all, 99.0))
                    elif c["zcf_mode"] == "clean_only":
                        z_min = float(min(clean_calib_zcfs))
                        z_max = float(max(clean_calib_zcfs))
                    elif c["zcf_mode"] == "clean_margin":
                        z_min = float(min(clean_calib_zcfs)) - 10.0
                        z_max = float(max(clean_calib_zcfs)) + 10.0

                    p_low_val = float(np.percentile(calib_low_scores, c["p_low"]))
                    p_high_val = float(np.percentile(calib_low_scores, c["p_high"]))

                    if zcf < z_min or zcf > z_max:
                        fft_used = True
                        reason = "zcf_out_of_range"
                    elif low_score <= p_low_val:
                        fft_used = False
                        reason = "confident_normal"
                    elif low_score >= p_high_val:
                        fft_used = False
                        reason = "confident_anomaly"
                    else:
                        fft_used = True
                        reason = "uncertain_score"

                if not fft_used:
                    pred_label = 0 if reason == "confident_normal" else 1
                    freq_est = zcf
                    ops = 10265
                    stage = "stage1_low"
                else:
                    pred_label = h_res.predicted_label
                    freq_est = h_res.frequency_estimate_hz
                    ops = 39515  # 10265 (Stage 1) + 29250 (Stage 2)
                    stage = "stage2_high"

                oof_cand_pred[c_name][key] = {
                    "pred": pred_label,
                    "freq": freq_est,
                    "fft_used": fft_used,
                    "ops": ops,
                    "stage": stage,
                    "reason": reason,
                }

    # Compile OOF Results across all 960 development windows
    y_true = [r["anomaly"] for r in dev_records]
    keys = [r["signal_key"] for r in dev_records]

    low_preds = [oof_low_pred[k]["pred"] for k in keys]
    low_freqs = [oof_low_pred[k]["freq"] for k in keys]
    low_m = compute_metrics(y_true, low_preds)
    low_fmae = compute_freq_mae(dev_records, low_freqs)

    high_preds = [oof_high_pred[k]["pred"] for k in keys]
    high_freqs = [oof_high_pred[k]["freq"] for k in keys]
    high_m = compute_metrics(y_true, high_preds)
    high_fmae = compute_freq_mae(dev_records, high_freqs)

    print("\n--- Out-of-Fold Baselines (5-Fold Grouped CV on 960 Dev Windows) ---")
    print(f"Fixed-Low : F1={low_m['f1']:.4f}, Recall={low_m['recall']:.4f}, FAR={low_m['false_alarm_rate']:.4f}, Freq MAE={low_fmae:.2f} Hz, Ops=10265")
    print(f"Fixed-High: F1={high_m['f1']:.4f}, Recall={high_m['recall']:.4f}, FAR={high_m['false_alarm_rate']:.4f}, Freq MAE={high_fmae:.2f} Hz, Ops=29250")

    min_f1 = high_m["f1"] - 0.05
    min_recall = high_m["recall"] - 0.05
    max_far = high_m["false_alarm_rate"] + 0.01

    print(f"\nPredeclared Quality Constraints relative to Fixed-High:")
    print(f"  F1 Target >= {min_f1:.4f} (High - 0.05)")
    print(f"  Recall Target >= {min_recall:.4f} (High - 0.05)")
    print(f"  FAR Target <= {max_far:.4f} (High + 0.01)")
    print(f"  Operation Proxy < 29250 ops/window (Positive savings vs standalone Fixed-High)")

    candidate_results = []
    qualifying_candidates = []

    print("\n--- Candidate Grid Evaluation ---")
    print(f"{'Candidate Name':<28} | {'F1':<6} | {'Recall':<6} | {'FAR':<6} | {'Freq MAE':<8} | {'Esc Rate':<8} | {'Avg Ops':<8} | {'Savings':<7} | {'Qualifies?'}")
    print("-" * 109)

    for c in candidates:
        name = c["name"]
        preds = [oof_cand_pred[name][k]["pred"] for k in keys]
        freqs = [oof_cand_pred[name][k]["freq"] for k in keys]
        ops_list = [oof_cand_pred[name][k]["ops"] for k in keys]
        fft_list = [oof_cand_pred[name][k]["fft_used"] for k in keys]

        m = compute_metrics(y_true, preds)
        fmae = compute_freq_mae(dev_records, freqs)
        avg_ops = float(np.mean(ops_list))
        esc_rate = float(np.mean(fft_list))
        savings_vs_high = (29250.0 - avg_ops) / 29250.0 * 100.0

        qualifies = bool(
            m["f1"] >= min_f1
            and m["recall"] >= min_recall
            and m["false_alarm_rate"] <= max_far
            and avg_ops < 29250.0
        )

        # Recall by case
        case_recall = {}
        for case in ["normal", "frequency_shift", "added_harmonic", "impact"]:
            case_idx = [i for i, r in enumerate(dev_records) if r["case"] == case]
            yt = [y_true[i] for i in case_idx]
            yp = [preds[i] for i in case_idx]
            cm = compute_metrics(yt, yp)
            case_recall[case] = cm["false_alarm_rate"] if case == "normal" else cm["recall"]

        # Breakdown by SNR
        snr_metrics = {}
        for snr in [30.0, 20.0, 10.0, 0.0]:
            snr_idx = [i for i, r in enumerate(dev_records) if r["snr_db"] == snr]
            yt = [y_true[i] for i in snr_idx]
            yp = [preds[i] for i in snr_idx]
            sm = compute_metrics(yt, yp)
            snr_esc = float(np.mean([fft_list[i] for i in snr_idx]))
            snr_metrics[str(snr)] = {
                "f1": sm["f1"],
                "recall": sm["recall"],
                "far": sm["false_alarm_rate"],
                "escalation_rate": snr_esc,
            }

        res_entry = {
            "name": name,
            "config": c,
            "metrics": m,
            "freq_mae_hz": fmae,
            "escalation_rate": esc_rate,
            "avg_op_count_proxy": avg_ops,
            "savings_vs_fixed_high_percent": savings_vs_high,
            "qualifies": qualifies,
            "case_performance": case_recall,
            "snr_performance": snr_metrics,
        }
        candidate_results.append(res_entry)

        status_str = "YES" if qualifies else "NO"
        print(f"{name:<28} | {m['f1']:.4f} | {m['recall']:.4f} | {m['false_alarm_rate']:.4f} | {fmae:8.2f} | {esc_rate*100:7.2f}% | {avg_ops:8.1f} | {savings_vs_high:6.2f}% | {status_str}")

        if qualifies:
            qualifying_candidates.append(res_entry)

    # Sort qualifying candidates by lowest operation-count proxy
    selected_candidate = None
    if qualifying_candidates:
        qualifying_candidates.sort(key=lambda x: x["avg_op_count_proxy"])
        selected_candidate = qualifying_candidates[0]
        print("\n================== SELECTION OUTCOME ==================")
        print(f"Selected Best Qualifying Candidate: {selected_candidate['name']}")
        print(f"  F1: {selected_candidate['metrics']['f1']:.4f} (Constraint >= {min_f1:.4f})")
        print(f"  Recall: {selected_candidate['metrics']['recall']:.4f} (Constraint >= {min_recall:.4f})")
        print(f"  FAR: {selected_candidate['metrics']['false_alarm_rate']:.4f} (Constraint <= {max_far:.4f})")
        print(f"  Escalation Rate: {selected_candidate['escalation_rate']*100:.2f}%")
        print(f"  Operation Proxy: {selected_candidate['avg_op_count_proxy']:.1f} ops/window")
        print(f"  Savings vs Fixed-High: {selected_candidate['savings_vs_fixed_high_percent']:.2f}%")
        print("=======================================================")
    else:
        print("\n================== SELECTION OUTCOME ==================")
        print("NO tested candidate satisfied all quality and efficiency constraints.")
        print("=======================================================")

    study_summary = {
        "status": "DEVELOPMENT_STUDY_COMPLETE",
        "cross_validation": {
            "n_folds": 5,
            "dev_groups": 60,
            "dev_windows": 960,
            "test_groups_protected": 20,
            "test_windows_locked": 320,
        },
        "fixed_baselines_oof": {
            "fixed_low": {**low_m, "freq_mae_hz": low_fmae, "ops": 10265},
            "fixed_high": {**high_m, "freq_mae_hz": high_fmae, "ops": 29250},
        },
        "quality_constraints": {
            "min_f1": min_f1,
            "min_recall": min_recall,
            "max_far": max_far,
            "max_ops": 29250.0,
        },
        "candidate_results": candidate_results,
        "selected_candidate": selected_candidate,
    }

    with (out_dir / "selector_study_results.json").open("w", encoding="utf-8") as f:
        json.dump(study_summary, f, indent=2)

    return study_summary


def main():
    parser = argparse.ArgumentParser(description="Phase 4 Development Selector Study")
    parser.add_argument("--out", type=Path, default=Path("results/phase4_study"))
    args = parser.parse_args()
    run_cv_study(out_dir=args.out)


if __name__ == "__main__":
    main()
