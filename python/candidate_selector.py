"""Candidate Adaptive Selector (v1.1 candidate): two-stage context-aware processing.

Preserves the Fixed-Low and Fixed-High baseline definitions.
Implements the revised decision cascade identified during the Phase 4 investigation:
1. Stage 1 (Fixed-Low): Evaluates time-domain features [log(RMS), log(crest), log(kurtosis)]
   and zero-crossing frequency. Cost: 10 * N + 25 = 10,265 ops.
2. Step 1: Confident Anomaly Check.
   If low_score >= p_anomaly_threshold (calibrated strictly above normal calibration distribution):
   Stop at Stage 1 as Confident Anomaly (reason: 'confident_anomaly').
   This prevents high-kurtosis impact signals from paying expensive FFT escalation overhead.
3. Step 2: Frequency Discrepancy Check.
   If zcf < z_min or zcf > z_max (clean-calibrated machine fundamental bounds):
   Escalate to Fixed-High FFT stage (reason: 'zcf_out_of_range').
   This captures clean frequency shifts and routes noise-corrupted signals to FFT.
4. Step 3: Confident Normal Check.
   Else if low_score <= p_normal_threshold (calibrated at lower percentile of normal calibration):
   Stop at Stage 1 as Confident Normal (reason: 'confident_normal').
5. Step 4: Uncertain Score Check.
   Else (low_score between p_normal and p_anomaly):
   Escalate to Fixed-High FFT stage (reason: 'uncertain_score').
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import time
import numpy as np

from baselines import FixedLowDetector, FixedHighDetector
from candidate_features import get_clean_calibrated_zcf_bounds
from features import zero_crossing_frequency


@dataclass
class CandidateResult:
    method: str  # 'candidate_adaptive_v1.1'
    predicted_label: int  # 0: normal, 1: anomaly
    final_anomaly_score: float
    frequency_estimate_hz: float | None
    stage_reached: str  # 'stage1_low' or 'stage2_high'
    fft_used: bool  # True if escalated to Fixed-High
    escalation_reason: str  # 'none', 'confident_anomaly', 'zcf_out_of_range', 'confident_normal', 'uncertain_score'
    low_stage_score: float
    high_stage_score: float | None
    execution_time_s: float
    op_count_proxy: int


class CandidateAdaptiveSelector:
    """Candidate adaptive selector combining Fixed-Low and Fixed-High baselines."""

    def __init__(
        self,
        fixed_low: FixedLowDetector,
        fixed_high: FixedHighDetector,
        sample_rate_hz: int = 4000,
        p_normal_percentile: float = 60.0,
        p_anomaly_multiplier: float = 1.2,
        snr_clean_threshold_db: float = 20.0,
        margin_hz: float = 10.0,
    ):
        self.fixed_low = fixed_low
        self.fixed_high = fixed_high
        self.sample_rate_hz = sample_rate_hz
        self.p_normal_percentile = p_normal_percentile
        self.p_anomaly_multiplier = p_anomaly_multiplier
        self.snr_clean_threshold_db = snr_clean_threshold_db
        self.margin_hz = margin_hz

        # Calibrated boundaries
        self.low_score_p_normal: float | None = None
        self.low_score_p_anomaly: float | None = None
        self.zcf_min: float | None = None
        self.zcf_max: float | None = None

    def calibrate_selector(
        self,
        normal_calib_signals: Sequence[np.ndarray],
        normal_calib_snrs: Sequence[float],
    ) -> dict[str, float]:
        """Derive selector boundaries from normal threshold-calibration windows only."""
        if len(normal_calib_signals) == 0:
            raise ValueError("Selector calibration requires at least one normal calibration signal")
        if self.fixed_low.threshold is None or self.fixed_high.threshold is None:
            raise RuntimeError("Both fixed baselines must be calibrated before calibrating selector")

        low_scores = [self.fixed_low.score_window(x) for x in normal_calib_signals]
        self.low_score_p_normal = float(np.percentile(low_scores, self.p_normal_percentile))
        self.low_score_p_anomaly = float(max(low_scores)) * self.p_anomaly_multiplier

        self.zcf_min, self.zcf_max = get_clean_calibrated_zcf_bounds(
            list(normal_calib_signals),
            list(normal_calib_snrs),
            sample_rate_hz=self.sample_rate_hz,
            snr_clean_threshold_db=self.snr_clean_threshold_db,
            margin_hz=self.margin_hz,
        )

        return {
            "low_score_p_normal": self.low_score_p_normal,
            "low_score_p_anomaly": self.low_score_p_anomaly,
            "zcf_min": self.zcf_min,
            "zcf_max": self.zcf_max,
        }

    def predict(self, x: np.ndarray) -> CandidateResult:
        """Process window using candidate policy with cascade accounting."""
        if (
            self.low_score_p_normal is None
            or self.low_score_p_anomaly is None
            or self.zcf_min is None
            or self.zcf_max is None
        ):
            raise RuntimeError("Candidate selector must be calibrated before prediction")

        t0 = time.perf_counter()
        n = len(x)
        stage1_ops = 10 * n + 25

        low_res = self.fixed_low.predict(x)
        low_score = low_res.anomaly_score
        zcf = low_res.frequency_estimate_hz

        # Decision cascade order:
        # 1. Obvious time-domain anomaly (e.g. impact) decided at Stage 1
        if low_score >= self.low_score_p_anomaly:
            t_elapsed = time.perf_counter() - t0
            return CandidateResult(
                method="candidate_adaptive_v1.1",
                predicted_label=1,
                final_anomaly_score=low_score,
                frequency_estimate_hz=zcf,
                stage_reached="stage1_low",
                fft_used=False,
                escalation_reason="confident_anomaly",
                low_stage_score=low_score,
                high_stage_score=None,
                execution_time_s=t_elapsed,
                op_count_proxy=stage1_ops,
            )

        # 2. Clean frequency out-of-range or high-noise frequency smearing -> Escalate
        if zcf < self.zcf_min or zcf > self.zcf_max:
            n_log2_n = int(n * np.log2(max(n, 2)))
            stage2_ops = int(2.5 * n_log2_n + 3.5 * n + 66)
            total_ops = stage1_ops + stage2_ops
            high_res = self.fixed_high.predict(x)
            t_elapsed = time.perf_counter() - t0
            return CandidateResult(
                method="candidate_adaptive_v1.1",
                predicted_label=high_res.predicted_label,
                final_anomaly_score=high_res.anomaly_score,
                frequency_estimate_hz=high_res.frequency_estimate_hz,
                stage_reached="stage2_high",
                fft_used=True,
                escalation_reason="zcf_out_of_range",
                low_stage_score=low_score,
                high_stage_score=high_res.anomaly_score,
                execution_time_s=t_elapsed,
                op_count_proxy=total_ops,
            )

        # 3. Confident normal low-score -> Stage 1 normal
        if low_score <= self.low_score_p_normal:
            t_elapsed = time.perf_counter() - t0
            return CandidateResult(
                method="candidate_adaptive_v1.1",
                predicted_label=0,
                final_anomaly_score=low_score,
                frequency_estimate_hz=zcf,
                stage_reached="stage1_low",
                fft_used=False,
                escalation_reason="confident_normal",
                low_stage_score=low_score,
                high_stage_score=None,
                execution_time_s=t_elapsed,
                op_count_proxy=stage1_ops,
            )

        # 4. Uncertain score between p_normal and p_anomaly -> Escalate to Fixed-High
        n_log2_n = int(n * np.log2(max(n, 2)))
        stage2_ops = int(2.5 * n_log2_n + 3.5 * n + 66)
        total_ops = stage1_ops + stage2_ops
        high_res = self.fixed_high.predict(x)
        t_elapsed = time.perf_counter() - t0
        return CandidateResult(
            method="candidate_adaptive_v1.1",
            predicted_label=high_res.predicted_label,
            final_anomaly_score=high_res.anomaly_score,
            frequency_estimate_hz=high_res.frequency_estimate_hz,
            stage_reached="stage2_high",
            fft_used=True,
            escalation_reason="uncertain_score",
            low_stage_score=low_score,
            high_stage_score=high_res.anomaly_score,
            execution_time_s=t_elapsed,
            op_count_proxy=total_ops,
        )
