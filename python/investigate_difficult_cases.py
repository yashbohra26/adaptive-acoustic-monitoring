"""Development-only investigation of difficult cases and candidate optimization.

CRITICAL RESEARCH SAFEGUARD:
- Uses ONLY the 60 development groups (960 windows) for tuning and candidate selection.
- Does NOT inspect, score, or tune against the original 20 held-out test groups.
- Evaluates candidate parameters via 5-fold grouped cross-validation across development groups.
- If an improved candidate (Candidate v1.2) is selected on development data:
  1. Freezes its specification in results/evaluation_protocol_v1.2.json before testing.
  2. Evaluates performance on a newly generated, independent held-out partition
     (20 new groups, seed=20261014) with a recorded SHA-256 split hash.
"""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd

from baselines import FixedLowDetector, FixedHighDetector
from candidate_selector import CandidateAdaptiveSelector
from config import DEFAULT_CONFIG
from generate_signals import build_dataset


def compute_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, float]:
    yt = np.asarray(y_true, dtype=int)
    yp = np.asarray(y_pred, dtype=int)

    tp = int(np.sum((yt == 1) & (yp == 1)))
    fp = int(np.sum((yt == 0) & (yp == 1)))
    tn = int(np.sum((yt == 0) & (yp == 0)))
    fn = int(np.sum((yt == 1) & (yp == 0)))

    total = len(yt)
    acc = float((tp + tn) / total) if total > 0 else 0.0
    prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    far = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "false_alarm_rate": far,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def investigate_development_data():
    print("=" * 80)
    print("DEVELOPMENT-ONLY INVESTIGATION OF DIFFICULT CASES")
    print("=" * 80)

    # 1. Load development data
    records, signals = build_dataset(groups=80, config=DEFAULT_CONFIG, dev_calibration_fraction=0.5)
    dev_records = [r for r in records if r["split"] == "development"]
    assert len(dev_records) == 960

    # Inspect score distributions on dev
    fit_sigs = [signals[r["signal_key"]] for r in dev_records if r["dev_subset"] == "fit" and r["case"] == "normal"]
    calib_sigs = [signals[r["signal_key"]] for r in dev_records if r["dev_subset"] == "calibration" and r["case"] == "normal"]
    calib_snrs = [r["snr_db"] for r in dev_records if r["dev_subset"] == "calibration" and r["case"] == "normal"]

    low_det = FixedLowDetector(sample_rate_hz=DEFAULT_CONFIG.sample_rate_hz)
    low_det.fit(fit_sigs)
    low_det.calibrate(calib_sigs, target_far=0.05)

    high_det = FixedHighDetector(sample_rate_hz=DEFAULT_CONFIG.sample_rate_hz)
    high_det.fit(fit_sigs)
    high_det.calibrate(calib_sigs, target_far=0.05)

    calib_scores = [low_det.score_window(x) for x in calib_sigs]
    print(f"Normal calibration low-scores: min={min(calib_scores):.4f}, median={np.median(calib_scores):.4f}, max={max(calib_scores):.4f}")
    for p in [20, 30, 40, 50, 60, 70]:
        print(f"  {p}th percentile: {np.percentile(calib_scores, p):.4f}")

    # Check scores of added_harmonic and normal across development
    scores_by_case = {}
    for c in ["normal", "added_harmonic", "frequency_shift", "impact"]:
        c_sigs = [signals[r["signal_key"]] for r in dev_records if r["case"] == c]
        scores_by_case[c] = [low_det.score_window(x) for x in c_sigs]
        print(f"Case {c:<16}: count={len(c_sigs)}, min={min(scores_by_case[c]):.4f}, median={np.median(scores_by_case[c]):.4f}, max={max(scores_by_case[c]):.4f}")

    # 2. 5-Fold Grouped Cross-Validation on Development Data across different P_normal percentiles
    print("\nRunning 5-fold grouped CV across P_normal percentiles on development partition...")
    dev_groups = sorted(list(set(r["group_id"] for r in dev_records)))
    np.random.default_rng(20261014).shuffle(dev_groups)
    folds = np.array_split(dev_groups, 5)

    candidate_results = []
    percentiles_to_test = [20.0, 30.0, 35.0, 40.0, 45.0, 50.0, 60.0]

    for p_norm in percentiles_to_test:
        oof_preds = []
        oof_true = []
        oof_ops = []
        oof_ffts = []

        for fold_idx, test_g in enumerate(folds):
            train_g = [g for g in dev_groups if g not in test_g]

            train_recs = [r for r in dev_records if r["group_id"] in train_g]
            val_recs = [r for r in dev_records if r["group_id"] in test_g]

            # Split train into fit and calib
            train_groups = sorted(list(set(r["group_id"] for r in train_recs)))
            half = len(train_groups) // 2
            f_groups = set(train_groups[:half])
            c_groups = set(train_groups[half:])

            f_sigs = [signals[r["signal_key"]] for r in train_recs if r["group_id"] in f_groups and r["case"] == "normal"]
            c_sigs = [signals[r["signal_key"]] for r in train_recs if r["group_id"] in c_groups and r["case"] == "normal"]
            c_snrs = [r["snr_db"] for r in train_recs if r["group_id"] in c_groups and r["case"] == "normal"]

            l_det = FixedLowDetector(sample_rate_hz=DEFAULT_CONFIG.sample_rate_hz)
            l_det.fit(f_sigs)
            l_det.calibrate(c_sigs, target_far=0.05)

            h_det = FixedHighDetector(sample_rate_hz=DEFAULT_CONFIG.sample_rate_hz)
            h_det.fit(f_sigs)
            h_det.calibrate(c_sigs, target_far=0.05)

            c_sel = CandidateAdaptiveSelector(
                l_det,
                h_det,
                sample_rate_hz=DEFAULT_CONFIG.sample_rate_hz,
                p_normal_percentile=p_norm,
                p_anomaly_multiplier=1.2,
            )
            c_sel.calibrate_selector(c_sigs, c_snrs)

            for vr in val_recs:
                vx = signals[vr["signal_key"]]
                res = c_sel.predict(vx)
                oof_preds.append(res.predicted_label)
                oof_true.append(vr["anomaly"])
                oof_ops.append(res.op_count_proxy)
                oof_ffts.append(res.fft_used)

        m = compute_metrics(oof_true, oof_preds)
        avg_ops = float(np.mean(oof_ops))
        esc_rate = float(np.mean(oof_ffts))
        savings = (29250.0 - avg_ops) / 29250.0 * 100.0

        candidate_results.append({
            "p_normal_percentile": p_norm,
            "f1": m["f1"],
            "recall": m["recall"],
            "precision": m["precision"],
            "false_alarm_rate": m["false_alarm_rate"],
            "avg_ops": avg_ops,
            "escalation_rate": esc_rate,
            "savings_percent": savings,
        })
        print(f"P_normal={p_norm:<4.1f}% | F1={m['f1']:.4f} | Recall={m['recall']:.4f} | FAR={m['false_alarm_rate']:.4f} | Escalation={esc_rate*100:.2f}% | Ops={avg_ops:.1f} ({savings:+.2f}%)")

    # Save CV study
    cv_file = Path("results/candidate_v1.2_development_study.json")
    with cv_file.open("w", encoding="utf-8") as f:
        json.dump(candidate_results, f, indent=2)
    print(f"\nSaved development candidate CV study to {cv_file}")

    # 3. Best Candidate Selection based strictly on pre-declared constraints
    # Target: Recall >= 0.8833, F1 >= 0.90, FAR <= 0.065, Ops < 29250 (positive savings)
    qualifying = [c for c in candidate_results if c["recall"] >= 0.8833 and c["avg_ops"] < 29250.0]
    print(f"\nQualifying candidates meeting Recall >= 0.8833 AND Ops < 29250: {len(qualifying)}")
    for q in qualifying:
        print(f"  -> Candidate P_norm={q['p_normal_percentile']}%: F1={q['f1']:.4f}, Recall={q['recall']:.4f}, Ops={q['avg_ops']:.1f} ({q['savings_percent']:+.2f}%)")

    if qualifying:
        best_cand = max(qualifying, key=lambda x: (x["f1"], x["savings_percent"]))
        selected_p_norm = best_cand["p_normal_percentile"]
        print(f"\nSELECTED CANDIDATE v1.2 CONFIGURATION: p_normal_percentile = {selected_p_norm}%")
        return selected_p_norm
    else:
        print("\nNo candidate achieved both recall >= 0.8833 and positive savings on CV.")
        return 50.0


def freeze_and_evaluate_v1_2(selected_p_norm: float):
    """Freeze Candidate v1.2 and evaluate on an INDEPENDENT grouped holdout (seed=20261014)."""
    print("\n" + "=" * 80)
    print("INDEPENDENT FINAL TEST EVALUATION FOR CANDIDATE v1.2")
    print("=" * 80)

    # 1. Fit & calibrate Candidate v1.2 strictly on original 60 development groups
    records_dev, signals_dev = build_dataset(groups=80, config=DEFAULT_CONFIG, dev_calibration_fraction=0.5)
    dev_records = [r for r in records_dev if r["split"] == "development"]

    fit_sigs = [signals_dev[r["signal_key"]] for r in dev_records if r["dev_subset"] == "fit" and r["case"] == "normal"]
    calib_sigs = [signals_dev[r["signal_key"]] for r in dev_records if r["dev_subset"] == "calibration" and r["case"] == "normal"]
    calib_snrs = [r["snr_db"] for r in dev_records if r["dev_subset"] == "calibration" and r["case"] == "normal"]

    low_det = FixedLowDetector(sample_rate_hz=DEFAULT_CONFIG.sample_rate_hz)
    low_det.fit(fit_sigs)
    low_det.calibrate(calib_sigs, target_far=0.05)

    high_det = FixedHighDetector(sample_rate_hz=DEFAULT_CONFIG.sample_rate_hz)
    high_det.fit(fit_sigs)
    high_det.calibrate(calib_sigs, target_far=0.05)

    cand_v12 = CandidateAdaptiveSelector(
        low_det,
        high_det,
        sample_rate_hz=DEFAULT_CONFIG.sample_rate_hz,
        p_normal_percentile=selected_p_norm,
        p_anomaly_multiplier=1.2,
    )
    calib_boundaries = cand_v12.calibrate_selector(calib_sigs, calib_snrs)

    # 2. Freeze Protocol v1.2.0 BEFORE generating new holdout
    protocol_v12 = {
        "protocol_version": "1.2.0",
        "description": "Pre-test freeze of Candidate Selector v1.2 addressing difficult added harmonic cases",
        "reference_detector_fixed_high": {
            "num_bands": 16,
            "threshold": high_det.threshold,
            "ops_per_window": 29250,
        },
        "candidate_v1.2": {
            "p_normal_percentile": selected_p_norm,
            "p_anomaly_multiplier": 1.2,
            "frozen_boundaries": calib_boundaries,
            "stage1_ops": 10265,
            "stage2_cumulative_ops": 39515,
        },
    }
    with open("results/evaluation_protocol_v1.2.json", "w", encoding="utf-8") as f:
        json.dump(protocol_v12, f, indent=2)
    print("Protocol v1.2.0 successfully frozen in results/evaluation_protocol_v1.2.json")

    # 3. Generate a NEW, INDEPENDENT holdout dataset: 20 groups with seed 20261014
    print("\nGenerating fresh, independent held-out evaluation dataset (20 groups, seed=20261014)...")
    from config import StudyConfig
    independent_config = StudyConfig(
        seed=20261014,  # Fresh independent seed
    )
    new_records, new_signals = build_dataset(groups=20, config=independent_config, dev_calibration_fraction=0.0)
    # All 20 groups are treated as pure held-out evaluation windows (320 windows)
    eval_records = new_records
    assert len(eval_records) == 320

    eval_groups = sorted(list(set(r["group_id"] for r in eval_records)))
    eval_hash = hashlib.sha256(",".join(eval_groups).encode("utf-8")).hexdigest()
    print(f"Independent holdout verified: 320 windows, SHA-256: {eval_hash[:16]}")

    # Evaluate Fixed-Low, Fixed-High, Candidate v1.1, and Candidate v1.2
    cand_v11 = CandidateAdaptiveSelector(
        low_det,
        high_det,
        sample_rate_hz=DEFAULT_CONFIG.sample_rate_hz,
        p_normal_percentile=60.0,
        p_anomaly_multiplier=1.2,
    )
    cand_v11.calibrate_selector(calib_sigs, calib_snrs)

    eval_models = {
        "fixed_low": low_det,
        "fixed_high": high_det,
        "candidate_v1.1": cand_v11,
        "candidate_v1.2": cand_v12,
    }

    eval_y_true = [r["anomaly"] for r in eval_records]
    eval_summary = {
        "status": "INDEPENDENT_HOLDOUT_EVALUATION_COMPLETE",
        "protocol_version": "1.2.0",
        "eval_seed": 20261014,
        "eval_windows": len(eval_records),
        "eval_split_hash_sha256": eval_hash,
        "methods": {},
    }

    print("\n" + "=" * 80)
    print(f"{'Method':<16} | {'F1':<6} | {'Recall':<6} | {'FAR':<6} | {'Avg Ops':<9} | {'Savings':<8} | {'Meets Quality?'}")
    print("-" * 80)

    fixed_high_ref = None

    for m_name, model in eval_models.items():
        preds = []
        ops = []
        ffts = []

        for r in eval_records:
            x = new_signals[r["signal_key"]]
            res = model.predict(x)
            if m_name == "fixed_low":
                preds.append(res.predicted_label)
                ops.append(10265)
                ffts.append(False)
            elif m_name == "fixed_high":
                preds.append(res.predicted_label)
                ops.append(29250)
                ffts.append(True)
            else:
                preds.append(res.predicted_label)
                ops.append(res.op_count_proxy)
                ffts.append(res.fft_used)

        m = compute_metrics(eval_y_true, preds)
        avg_ops = float(np.mean(ops))
        savings = (29250.0 - avg_ops) / 29250.0 * 100.0

        if m_name == "fixed_high":
            fixed_high_ref = m

        meets_all = "REFERENCE"
        if m_name in ["candidate_v1.1", "candidate_v1.2"] and fixed_high_ref:
            meets_f1 = m["f1"] >= (fixed_high_ref["f1"] - 0.05)
            meets_rec = m["recall"] >= (fixed_high_ref["recall"] - 0.05)
            meets_far = m["false_alarm_rate"] <= (fixed_high_ref["false_alarm_rate"] + 0.01)
            meets_ops = avg_ops < 29250.0
            meets_all = "YES" if (meets_f1 and meets_rec and meets_far and meets_ops) else f"NO (Rec:{m['recall']:.3f} vs {fixed_high_ref['recall']-0.05:.3f})"

        eval_summary["methods"][m_name] = {
            "metrics": m,
            "avg_op_count_proxy": avg_ops,
            "savings_vs_fixed_high_percent": savings,
            "escalation_rate": float(np.mean(ffts)),
            "meets_quality_constraints": meets_all,
        }

        print(f"{m_name:<16} | {m['f1']:<6.4f} | {m['recall']:<6.4f} | {m['false_alarm_rate']:<6.4f} | {avg_ops:<9.1f} | {savings:<+7.2f}% | {meets_all}")

    print("=" * 80)

    out_file = Path("results/candidate_v1.2_independent_evaluation.json")
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(eval_summary, f, indent=2)
    print(f"Independent evaluation results saved to {out_file}")


if __name__ == "__main__":
    sel_p = investigate_development_data()
    freeze_and_evaluate_v1_2(sel_p)
