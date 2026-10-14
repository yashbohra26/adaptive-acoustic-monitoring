"""Provenance and uncertainty verification for the Candidate v1.2 independent evaluation.

RESEARCH SAFEGUARDS:
1. Does NOT overwrite or alter results/candidate_v1.2_independent_evaluation.json.
2. Documents complete signal-generation configuration, seed (20261014), group IDs,
   and label composition.
3. Derives BOTH:
   - eval_legacy_group_names_hash_sha256 (preserves exact historical hash)
   - eval_canonical_content_hash_sha256 (canonicalized floating-point signal data + metadata)
4. Computes 1,000-iteration group-level bootstrap confidence intervals (95% CI)
   by resampling entire source groups together.
5. Saves the verified provenance and bootstrap uncertainty to:
   results/candidate_v1.2_provenance_and_uncertainty.json
   and per-window reproduction predictions to:
   results/candidate_v1.2_reproduced_windows.csv.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Sequence
import numpy as np
import pandas as pd

from baselines import FixedLowDetector, FixedHighDetector
from candidate_selector import CandidateAdaptiveSelector
from config import StudyConfig, DEFAULT_CONFIG
from generate_signals import build_dataset


def compute_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> dict[str, float | int]:
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


def compute_canonical_content_hash(records: Sequence[dict], signals: dict[str, np.ndarray]) -> str:
    """Compute deterministic SHA-256 hash over canonical signal data and metadata."""
    sorted_recs = sorted(records, key=lambda r: (r["group_id"], r["case"], r["snr_db"]))
    hasher = hashlib.sha256()

    for r in sorted_recs:
        sig = signals[r["signal_key"]]
        meta_str = f"{r['group_id']}|{r['case']}|{r['snr_db']:.1f}|{r['anomaly']}|{r['target_frequency_hz']}"
        hasher.update(meta_str.encode("utf-8"))
        # Hash 8-decimal quantized float buffer to avoid platform float representation quirks
        quantized_bytes = np.round(sig, 8).tobytes()
        hasher.update(quantized_bytes)

    return hasher.hexdigest()


def run_group_bootstrap(
    records: Sequence[dict],
    predictions: dict[str, list[dict]],
    n_boot: int = 1000,
    seed: int = 20261014,
) -> dict[str, dict[str, list[float]]]:
    """Execute group-level bootstrap resampling across the 20 source groups."""
    rng = np.random.default_rng(seed)
    unique_groups = sorted(list(set(r["group_id"] for r in records)))
    n_groups = len(unique_groups)

    group_to_indices = {g: [] for g in unique_groups}
    for idx, r in enumerate(records):
        group_to_indices[r["group_id"]].append(idx)

    methods = list(predictions.keys())
    boot_stats = {m: {"f1": [], "recall": [], "precision": [], "false_alarm_rate": [], "avg_ops": [], "savings": []} for m in methods}

    fixed_high_ops = 29250.0

    for _ in range(n_boot):
        sampled_groups = rng.choice(unique_groups, size=n_groups, replace=True)
        boot_idx = []
        for g in sampled_groups:
            boot_idx.extend(group_to_indices[g])

        boot_recs = [records[i] for i in boot_idx]
        y_true = [boot_recs[i]["anomaly"] for i in range(len(boot_recs))]

        for m in methods:
            preds_m = [predictions[m][i] for i in boot_idx]
            yp = [p["pred"] for p in preds_m]
            ops = [p["ops"] for p in preds_m]

            m_metrics = compute_metrics(y_true, yp)
            m_avg_ops = float(np.mean(ops))
            m_savings = (fixed_high_ops - m_avg_ops) / fixed_high_ops * 100.0

            boot_stats[m]["f1"].append(m_metrics["f1"])
            boot_stats[m]["recall"].append(m_metrics["recall"])
            boot_stats[m]["precision"].append(m_metrics["precision"])
            boot_stats[m]["false_alarm_rate"].append(m_metrics["false_alarm_rate"])
            boot_stats[m]["avg_ops"].append(m_avg_ops)
            boot_stats[m]["savings"].append(m_savings)

    ci_results = {}
    for m in methods:
        ci_results[m] = {}
        for metric, values in boot_stats[m].items():
            ci_results[m][metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "ci_95": [
                    float(np.percentile(values, 2.5)),
                    float(np.percentile(values, 97.5)),
                ],
            }

    return ci_results


def verify_and_record_provenance():
    print("=" * 80)
    print("VERIFYING PROVENANCE & UNCERTAINTY FOR CANDIDATE v1.2 EVALUATION")
    print("=" * 80)

    # 1. Calibrate detectors on the 60 development groups (Seed 20261013)
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

    # Frozen Protocol v1.2 parameters
    with open("results/evaluation_protocol_v1.2.json", "r", encoding="utf-8") as f:
        proto_v12 = json.load(f)

    p_norm = proto_v12["candidate_v1.2"]["p_normal_percentile"]
    p_anom = proto_v12["candidate_v1.2"]["p_anomaly_multiplier"]

    cand_v12 = CandidateAdaptiveSelector(
        low_det,
        high_det,
        sample_rate_hz=DEFAULT_CONFIG.sample_rate_hz,
        p_normal_percentile=p_norm,
        p_anomaly_multiplier=p_anom,
    )
    cand_v12.calibrate_selector(calib_sigs, calib_snrs)

    cand_v11 = CandidateAdaptiveSelector(
        low_det,
        high_det,
        sample_rate_hz=DEFAULT_CONFIG.sample_rate_hz,
        p_normal_percentile=60.0,
        p_anomaly_multiplier=1.2,
    )
    cand_v11.calibrate_selector(calib_sigs, calib_snrs)

    # 2. Generate Independent Evaluation Dataset (20 groups, seed=20261014)
    eval_config = StudyConfig(seed=20261014)
    eval_records, eval_signals = build_dataset(groups=20, config=eval_config, dev_calibration_fraction=0.0)
    assert len(eval_records) == 320

    # 3. Derive Hashes
    eval_groups = sorted(list(set(r["group_id"] for r in eval_records)))
    legacy_group_hash = hashlib.sha256(",".join(eval_groups).encode("utf-8")).hexdigest()
    canonical_content_hash = compute_canonical_content_hash(eval_records, eval_signals)

    print(f"Legacy Group-Names Hash : {legacy_group_hash}")
    print(f"Canonical Content Hash  : {canonical_content_hash}")

    # Confirm legacy hash matches frozen artifact
    with open("results/candidate_v1.2_independent_evaluation.json", "r", encoding="utf-8") as f:
        frozen_data = json.load(f)
    assert frozen_data["eval_split_hash_sha256"] == legacy_group_hash, "Legacy group hash mismatch!"

    # 4. Generate Predictions and Verify Parity with Frozen Artifact
    models = {
        "fixed_low": low_det,
        "fixed_high": high_det,
        "candidate_v1.1": cand_v11,
        "candidate_v1.2": cand_v12,
    }

    predictions = {m: [] for m in models}
    window_rows = []

    y_true = [r["anomaly"] for r in eval_records]

    for idx, r in enumerate(eval_records):
        x = eval_signals[r["signal_key"]]
        row = {
            "window_idx": idx,
            "group_id": r["group_id"],
            "case": r["case"],
            "snr_db": r["snr_db"],
            "anomaly_true": r["anomaly"],
            "target_freq_hz": r["target_frequency_hz"],
        }

        for m_name, model in models.items():
            res = model.predict(x)
            if m_name == "fixed_low":
                pred = res.predicted_label
                ops = 10265
                fft_used = False
                stage = "stage1_low"
                reason = "none"
            elif m_name == "fixed_high":
                pred = res.predicted_label
                ops = 29250
                fft_used = True
                stage = "stage2_high"
                reason = "none"
            else:
                pred = res.predicted_label
                ops = res.op_count_proxy
                fft_used = res.fft_used
                stage = res.stage_reached
                reason = res.escalation_reason

            predictions[m_name].append({
                "pred": pred,
                "ops": ops,
                "fft_used": fft_used,
                "stage": stage,
                "reason": reason,
            })

            row[f"{m_name}_pred"] = pred
            row[f"{m_name}_ops"] = ops
            row[f"{m_name}_fft"] = fft_used
            row[f"{m_name}_stage"] = stage
            row[f"{m_name}_reason"] = reason

        window_rows.append(row)

    # Check metrics match frozen artifact 100%
    for m_name in models:
        yp = [p["pred"] for p in predictions[m_name]]
        m_calc = compute_metrics(y_true, yp)
        m_froz = frozen_data["methods"][m_name]["metrics"]

        assert m_calc["f1"] == m_froz["f1"], f"F1 mismatch for {m_name}"
        assert m_calc["recall"] == m_froz["recall"], f"Recall mismatch for {m_name}"
        assert m_calc["false_alarm_rate"] == m_froz["false_alarm_rate"], f"FAR mismatch for {m_name}"
        assert m_calc["tp"] == m_froz["tp"], f"TP mismatch for {m_name}"
        assert m_calc["fn"] == m_froz["fn"], f"FN mismatch for {m_name}"

    print("Verified 100% reproduction parity with frozen candidate_v1.2_independent_evaluation.json.")

    # Save reproduction window table
    df_reprod = pd.DataFrame(window_rows)
    df_reprod.to_csv("results/candidate_v1.2_reproduced_windows.csv", index=False)
    print("Saved reproduction window predictions to results/candidate_v1.2_reproduced_windows.csv")

    # 5. Group-Level Bootstrap Uncertainty Estimation (1,000 iterations)
    print("Running 1,000 group-level bootstrap resampling iterations for uncertainty estimation...")
    bootstrap_results = run_group_bootstrap(eval_records, predictions, n_boot=1000, seed=20261014)

    # 6. Compose Comprehensive Provenance Record
    provenance_record = {
        "record_type": "CANDIDATE_V1.2_PROVENANCE_AND_BOOTSTRAP_UNCERTAINTY",
        "description": "Comprehensive provenance, canonical data hashing, and bootstrap confidence intervals for Candidate v1.2 independent holdout evaluation.",
        "historical_frozen_artifact": "results/candidate_v1.2_independent_evaluation.json",
        "reproduction_windows_artifact": "results/candidate_v1.2_reproduced_windows.csv",
        "signal_generation": {
            "dataset_purpose": "Independent held-out evaluation for Candidate v1.2",
            "seed": 20261014,
            "sample_rate_hz": eval_config.sample_rate_hz,
            "duration_s": eval_config.duration_s,
            "samples_per_window": eval_config.samples_per_window,
            "snr_db_levels": list(eval_config.snr_db_levels),
            "normal_frequency_range_hz": list(eval_config.normal_frequency_range_hz),
            "shift_ratio_range": list(eval_config.shift_ratio_range),
            "impact_decay_s": eval_config.impact_decay_s,
            "impact_carrier_range_hz": list(eval_config.impact_carrier_range_hz),
            "total_windows": len(eval_records),
            "total_groups": len(eval_groups),
            "group_ids": eval_groups,
            "composition_by_case": {
                "normal": int(sum(1 for r in eval_records if r["case"] == "normal")),
                "frequency_shift": int(sum(1 for r in eval_records if r["case"] == "frequency_shift")),
                "added_harmonic": int(sum(1 for r in eval_records if r["case"] == "added_harmonic")),
                "impact": int(sum(1 for r in eval_records if r["case"] == "impact")),
            },
            "composition_by_snr": {
                str(s): int(sum(1 for r in eval_records if r["snr_db"] == s)) for s in eval_config.snr_db_levels
            },
            "eval_legacy_group_names_hash_sha256": legacy_group_hash,
            "eval_canonical_content_hash_sha256": canonical_content_hash,
        },
        "model_parameters": {
            "protocol_version": "1.2.0",
            "fixed_high_threshold": high_det.threshold,
            "candidate_v1.2_boundaries": proto_v12["candidate_v1.2"]["frozen_boundaries"],
        },
        "bootstrap_uncertainty_95ci": bootstrap_results,
        "scientific_disclaimers": [
            "All signals are synthetically generated under controlled Python waveform definitions.",
            "Group-level bootstrap confidence intervals characterize sampling variability across the 20 synthetic source groups; they do not constitute real-world industrial machine validation.",
            "Operation counts represent an algorithmic proxy; they do not represent physical microcontroller (ESP32) electrical power or battery life.",
            "Candidate v1.2 results are evaluated on an independent holdout (seed 20261014) and are not combined with Phase 5 held-out test data (seed 20261013).",
        ],
    }

    out_prov = Path("results/candidate_v1.2_provenance_and_uncertainty.json")
    with out_prov.open("w", encoding="utf-8") as f:
        json.dump(provenance_record, f, indent=2)

    print(f"\nSaved comprehensive provenance and uncertainty record to {out_prov}")

    # Print summary table of Bootstrap 95% CIs
    print("\n" + "=" * 80)
    print("INDEPENDENT EVALUATION METRICS WITH 95% BOOTSTRAP CONFIDENCE INTERVALS")
    print("=" * 80)
    for m in ["fixed_high", "candidate_v1.1", "candidate_v1.2"]:
        b = bootstrap_results[m]
        print(f"Method: {m}")
        print(f"  F1 Score : {b['f1']['mean']:.4f} (95% CI: [{b['f1']['ci_95'][0]:.4f}, {b['f1']['ci_95'][1]:.4f}])")
        print(f"  Recall   : {b['recall']['mean']:.4f} (95% CI: [{b['recall']['ci_95'][0]:.4f}, {b['recall']['ci_95'][1]:.4f}])")
        print(f"  FAR      : {b['false_alarm_rate']['mean']:.4f} (95% CI: [{b['false_alarm_rate']['ci_95'][0]:.4f}, {b['false_alarm_rate']['ci_95'][1]:.4f}])")
        print(f"  Avg Ops  : {b['avg_ops']['mean']:.1f} (95% CI: [{b['avg_ops']['ci_95'][0]:.1f}, {b['avg_ops']['ci_95'][1]:.1f}])")
        print(f"  Savings  : {b['savings']['mean']:+.2f}% (95% CI: [{b['savings']['ci_95'][0]:+.2f}%, {b['savings']['ci_95'][1]:+.2f}%])")
    print("=" * 80)


if __name__ == "__main__":
    verify_and_record_provenance()
