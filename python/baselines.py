"""Explicit, interpretable baseline detectors: Fixed-Low and Fixed-High.

Neither feature extraction nor FFT alone constitutes an anomaly detector.
This module implements:
1. Feature extraction and normal reference parameter estimation (median, robust scale/MAD)
   using ONLY normal model-fit windows.
2. Distance-from-normal anomaly score calculation.
3. Threshold calibration targeting a specified calibration false-alarm rate
   using ONLY normal threshold-calibration windows.
4. Window-level classification returning anomaly score, predicted label,
   and frequency estimate where meaningful.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import numpy as np

from features import (
    rms,
    crest_factor,
    kurtosis,
    zero_crossing_frequency,
    fft_peak_frequency,
    normalized_band_energies,
)


def compute_median_mad(
    values: np.ndarray,
    floor: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute column-wise median and robust scale (MAD * 1.4826) with a floor.

    For a 1D or 2D array of features, returns (median, scale).
    """
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[:, np.newaxis]

    med = np.median(values, axis=0)
    # Median Absolute Deviation
    abs_dev = np.abs(values - med)
    mad = np.median(abs_dev, axis=0)
    # Scale estimate for normal distribution: 1.4826 * MAD
    scale = 1.4826 * mad
    scale = np.maximum(scale, floor)
    return med, scale


def standardized_distance(
    features: np.ndarray,
    reference_median: np.ndarray,
    reference_scale: np.ndarray,
) -> np.ndarray:
    """Calculate Euclidean distance of robust z-scores (standardized feature deviations).

    distance = sqrt(sum(((x_i - med_i) / scale_i) ** 2))
    """
    features = np.asarray(features, dtype=float)
    if features.ndim == 1:
        z = (features - reference_median) / reference_scale
        return float(np.sqrt(np.sum(z**2)))
    else:
        z = (features - reference_median) / reference_scale
        return np.sqrt(np.sum(z**2, axis=-1))


def calibrate_threshold(
    calibration_scores: Sequence[float] | np.ndarray,
    target_far: float = 0.05,
) -> float:
    """Calibrate decision threshold to target a specific false-alarm rate on normal scores.

    If target_far is 0.05 (5%), the threshold is the (1.0 - target_far) percentile,
    i.e., the 95th percentile of normal calibration anomaly scores.
    """
    scores = np.asarray(calibration_scores, dtype=float)
    if len(scores) == 0:
        raise ValueError("Cannot calibrate threshold on empty scores")
    percentile = (1.0 - target_far) * 100.0
    # Use method='higher' or 'linear'. With numpy default linear interpolation:
    threshold = float(np.percentile(scores, percentile))
    return threshold


@dataclass
class BaselineResult:
    method: str
    anomaly_score: float
    predicted_label: int  # 0: normal, 1: anomaly
    threshold: float
    frequency_estimate_hz: float | None


class FixedLowDetector:
    """Fixed-low baseline using inexpensive time-domain features.

    Features: [log(RMS), log(crest_factor), log(kurtosis)]
    Frequency estimator: zero_crossing_frequency
    """

    def __init__(self, sample_rate_hz: int = 4000, scale_floor: float = 1e-4):
        self.sample_rate_hz = sample_rate_hz
        self.scale_floor = scale_floor
        self.reference_median: np.ndarray | None = None
        self.reference_scale: np.ndarray | None = None
        self.threshold: float | None = None
        self.calibrated_far: float | None = None

    @staticmethod
    def extract_vector(x: np.ndarray) -> np.ndarray:
        """Extract [log(RMS), log(crest_factor), log(kurtosis)]."""
        r = rms(x)
        cf = crest_factor(x)
        k = kurtosis(x)
        # Safe log transforms with small floors
        return np.array([
            np.log(max(r, 1e-8)),
            np.log(max(cf, 1e-8)),
            np.log(max(k, 1e-8)),
        ], dtype=float)

    def fit(self, normal_fit_signals: Sequence[np.ndarray]) -> None:
        """Fit reference median and scale on normal model-fit windows only."""
        if len(normal_fit_signals) == 0:
            raise ValueError("Fitting requires at least one normal signal")
        feature_matrix = np.array([self.extract_vector(x) for x in normal_fit_signals])
        self.reference_median, self.reference_scale = compute_median_mad(
            feature_matrix, floor=self.scale_floor
        )

    def score_window(self, x: np.ndarray) -> float:
        """Compute distance-from-normal score."""
        if self.reference_median is None or self.reference_scale is None:
            raise RuntimeError("Detector must be fitted before scoring")
        vec = self.extract_vector(x)
        return float(standardized_distance(vec, self.reference_median, self.reference_scale))

    def calibrate(
        self,
        normal_calib_signals: Sequence[np.ndarray],
        target_far: float = 0.05,
    ) -> float:
        """Set decision threshold on normal threshold-calibration windows only."""
        if self.reference_median is None:
            raise RuntimeError("Detector must be fitted before calibration")
        scores = [self.score_window(x) for x in normal_calib_signals]
        self.threshold = calibrate_threshold(scores, target_far=target_far)
        self.calibrated_far = target_far
        return self.threshold

    def predict(self, x: np.ndarray) -> BaselineResult:
        """Classify a window and return anomaly score, label, and zero-crossing frequency."""
        if self.threshold is None:
            raise RuntimeError("Detector must be calibrated before prediction")
        score = self.score_window(x)
        pred = int(score > self.threshold)
        f_est = zero_crossing_frequency(x, self.sample_rate_hz)
        return BaselineResult(
            method="fixed_low",
            anomaly_score=score,
            predicted_label=pred,
            threshold=self.threshold,
            frequency_estimate_hz=f_est,
        )


class FixedHighDetector:
    """Fixed-high baseline using FFT-based normalized band energies (16 bands, 0-1200 Hz).

    Features: 16-dimensional normalized band energy vector.
    Frequency estimator: fft_peak_frequency
    """

    def __init__(
        self,
        sample_rate_hz: int = 4000,
        freq_min_hz: float = 0.0,
        freq_max_hz: float = 1200.0,
        num_bands: int = 16,
        scale_floor: float = 1e-4,
    ):
        self.sample_rate_hz = sample_rate_hz
        self.freq_min_hz = freq_min_hz
        self.freq_max_hz = freq_max_hz
        self.num_bands = num_bands
        self.scale_floor = scale_floor
        self.reference_median: np.ndarray | None = None
        self.reference_scale: np.ndarray | None = None
        self.threshold: float | None = None
        self.calibrated_far: float | None = None

    def extract_vector(self, x: np.ndarray) -> np.ndarray:
        """Extract 16 normalized band energies."""
        return normalized_band_energies(
            x,
            self.sample_rate_hz,
            freq_min_hz=self.freq_min_hz,
            freq_max_hz=self.freq_max_hz,
            num_bands=self.num_bands,
        )

    def fit(self, normal_fit_signals: Sequence[np.ndarray]) -> None:
        """Fit reference median and scale on normal model-fit windows only."""
        if len(normal_fit_signals) == 0:
            raise ValueError("Fitting requires at least one normal signal")
        feature_matrix = np.array([self.extract_vector(x) for x in normal_fit_signals])
        self.reference_median, self.reference_scale = compute_median_mad(
            feature_matrix, floor=self.scale_floor
        )

    def score_window(self, x: np.ndarray) -> float:
        """Compute distance-from-normal score."""
        if self.reference_median is None or self.reference_scale is None:
            raise RuntimeError("Detector must be fitted before scoring")
        vec = self.extract_vector(x)
        return float(standardized_distance(vec, self.reference_median, self.reference_scale))

    def calibrate(
        self,
        normal_calib_signals: Sequence[np.ndarray],
        target_far: float = 0.05,
    ) -> float:
        """Set decision threshold on normal threshold-calibration windows only."""
        if self.reference_median is None:
            raise RuntimeError("Detector must be fitted before calibration")
        scores = [self.score_window(x) for x in normal_calib_signals]
        self.threshold = calibrate_threshold(scores, target_far=target_far)
        self.calibrated_far = target_far
        return self.threshold

    def predict(self, x: np.ndarray) -> BaselineResult:
        """Classify a window and return anomaly score, label, and FFT peak frequency."""
        if self.threshold is None:
            raise RuntimeError("Detector must be calibrated before prediction")
        score = self.score_window(x)
        pred = int(score > self.threshold)
        f_est = fft_peak_frequency(x, self.sample_rate_hz)
        return BaselineResult(
            method="fixed_high",
            anomaly_score=score,
            predicted_label=pred,
            threshold=self.threshold,
            frequency_estimate_hz=f_est,
        )
