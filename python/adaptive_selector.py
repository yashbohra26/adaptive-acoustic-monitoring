"""Adaptive Selector: two-stage context-aware processing combining Fixed-Low and Fixed-High.

The adaptive selector:
1. Runs Fixed-Low (inexpensive time-domain features) first.
2. Evaluates the low-stage anomaly score and zero-crossing frequency against
   boundaries derived EXCLUSIVELY from normal threshold-calibration windows:
   - Low-score 90th percentile (low_score_p90)
   - Low-score 99th percentile (low_score_p99)
   - Zero-crossing frequency 1st and 99th percentiles (zcf_p01, zcf_p99)
3. Decision Logic:
   - If zcf is outside [zcf_p01, zcf_p99]:
     Escalate to Fixed-High FFT stage (reason: 'zcf_out_of_range').
   - Else if low_score <= low_score_p90:
     Stop at Stage 1 as Confidently Normal (reason: 'confident_normal').
   - Else if low_score >= low_score_p99:
     Stop at Stage 1 as Confidently Anomalous (reason: 'confident_anomaly').
   - Else (low_score between p90 and p99):
     Escalate to Fixed-High FFT stage (reason: 'uncertain_score').
4. If escalated, the final prediction and frequency estimate come from the Fixed-High detector.
5. Work accounting:
   Records stage reached ('stage1_low' or 'stage2_high'), boolean fft_used,
   escalation reason, operation count proxy, and Python execution time.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import time
import numpy as np

from baselines import FixedLowDetector, FixedHighDetector, BaselineResult
from features import zero_crossing_frequency


@dataclass
class AdaptiveResult:
    method: str  # 'adaptive'
    predicted_label: int  # 0: normal, 1: anomaly
    final_anomaly_score: float  # score from the deciding stage
    frequency_estimate_hz: float | None
    stage_reached: str  # 'stage1_low' or 'stage2_high'
    fft_used: bool  # True if escalated to Fixed-High
    escalation_reason: str  # 'none', 'zcf_out_of_range', 'uncertain_score'
    low_stage_score: float
    high_stage_score: float | None
    execution_time_s: float
    op_count_proxy: int


class AdaptiveSelector:
    """Adaptive two-stage detector combining Fixed-Low and Fixed-High baselines."""

    def __init__(
        self,
        fixed_low: FixedLowDetector,
        fixed_high: FixedHighDetector,
        sample_rate_hz: int = 4000,
    ):
        self.fixed_low = fixed_low
        self.fixed_high = fixed_high
        self.sample_rate_hz = sample_rate_hz

        # Selector boundaries calibrated from normal calibration data
        self.low_score_p90: float | None = None
        self.low_score_p99: float | None = None
        self.zcf_p01: float | None = None
        self.zcf_p99: float | None = None

    def calibrate_selector(
        self,
        normal_calib_signals: Sequence[np.ndarray],
    ) -> dict[str, float]:
        """Derive selector boundaries from normal threshold-calibration windows only."""
        if len(normal_calib_signals) == 0:
            raise ValueError("Selector calibration requires at least one normal calibration signal")
        if self.fixed_low.threshold is None or self.fixed_high.threshold is None:
            raise RuntimeError("Both fixed baselines must be calibrated before calibrating selector")

        low_scores = [self.fixed_low.score_window(x) for x in normal_calib_signals]
        zcfs = [zero_crossing_frequency(x, self.sample_rate_hz) for x in normal_calib_signals]

        self.low_score_p90 = float(np.percentile(low_scores, 90.0))
        self.low_score_p99 = float(np.percentile(low_scores, 99.0))
        self.zcf_p01 = float(np.percentile(zcfs, 1.0))
        self.zcf_p99 = float(np.percentile(zcfs, 99.0))

        return {
            "low_score_p90": self.low_score_p90,
            "low_score_p99": self.low_score_p99,
            "zcf_p01": self.zcf_p01,
            "zcf_p99": self.zcf_p99,
        }

    def predict(self, x: np.ndarray) -> AdaptiveResult:
        """Process window using adaptive policy with stage recording and work accounting."""
        if (
            self.low_score_p90 is None
            or self.low_score_p99 is None
            or self.zcf_p01 is None
            or self.zcf_p99 is None
        ):
            raise RuntimeError("Selector must be calibrated before prediction")

        t0 = time.perf_counter()
        n = len(x)

        # Stage 1: Fixed-Low feature extraction and zero-crossing frequency
        # Operation count proxy:
        # RMS: ~2*N ops (multiply + accumulate + mean/sqrt)
        # Centering for Crest & Kurtosis: ~2*N ops
        # Crest Factor: ~N ops (max absolute)
        # Kurtosis: ~3*N ops (4th power + sum)
        # Zero-crossing: ~2*N ops (sign check + diff count)
        # Distance calculation: ~15 ops
        # Selector comparison: ~10 ops
        # Total Stage 1 ops ~= 10 * N + 25
        stage1_ops = 10 * n + 25

        low_res = self.fixed_low.predict(x)
        low_score = low_res.anomaly_score
        zcf = low_res.frequency_estimate_hz

        escalate = False
        reason = "none"

        # Check cheap zero-crossing frequency boundary
        if zcf < self.zcf_p01 or zcf > self.zcf_p99:
            escalate = True
            reason = "zcf_out_of_range"
        elif low_score <= self.low_score_p90:
            # Confidently normal
            escalate = False
            reason = "confident_normal"
        elif low_score >= self.low_score_p99:
            # Confidently anomaly
            escalate = False
            reason = "confident_anomaly"
        else:
            # Ambiguous low score between p90 and p99
            escalate = True
            reason = "uncertain_score"

        if not escalate:
            t_elapsed = time.perf_counter() - t0
            # Label comes from the low stage decision rule (score > threshold)
            # or directly from confidence region (normal for <= p90, anomaly for >= p99)
            pred_label = 0 if reason == "confident_normal" else 1

            return AdaptiveResult(
                method="adaptive",
                predicted_label=pred_label,
                final_anomaly_score=low_score,
                frequency_estimate_hz=zcf,
                stage_reached="stage1_low",
                fft_used=False,
                escalation_reason=reason,
                low_stage_score=low_score,
                high_stage_score=None,
                execution_time_s=t_elapsed,
                op_count_proxy=stage1_ops,
            )
        else:
            # Stage 2: Escalate to Fixed-High
            # Fixed-High Ops proxy:
            # Windowing: N ops
            # rFFT: ~2.5 * N * log2(N)
            # Magnitude squared / power: N ops
            # 16-band energy accumulation: N ops
            # Normalization: 16 ops
            # Standardized distance: ~50 ops
            # Peak search: N/2 ops
            # Total Stage 2 added ops ~= 2.5 * N * log2(N) + 3.5 * N + 66
            n_log2_n = int(n * np.log2(max(n, 2)))
            stage2_ops = int(2.5 * n_log2_n + 3.5 * n + 66)
            total_ops = stage1_ops + stage2_ops

            high_res = self.fixed_high.predict(x)
            t_elapsed = time.perf_counter() - t0

            return AdaptiveResult(
                method="adaptive",
                predicted_label=high_res.predicted_label,
                final_anomaly_score=high_res.anomaly_score,
                frequency_estimate_hz=high_res.frequency_estimate_hz,
                stage_reached="stage2_high",
                fft_used=True,
                escalation_reason=reason,
                low_stage_score=low_score,
                high_stage_score=high_res.anomaly_score,
                execution_time_s=t_elapsed,
                op_count_proxy=total_ops,
            )
