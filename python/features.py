"""Small, explicitly defined signal features; not classifiers."""
from __future__ import annotations

import numpy as np


def rms(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.sqrt(np.mean(x * x)))


def crest_factor(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    centered = x - np.mean(x)
    sd = float(np.sqrt(np.mean(centered * centered)))
    if sd <= 1e-12:
        return 0.0
    return float(np.max(np.abs(centered)) / sd)


def kurtosis(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    centered = x - np.mean(x)
    variance = float(np.mean(centered**2))
    if variance <= 1e-24:
        return 0.0
    return float(np.mean(centered**4) / variance**2)


def zero_crossing_frequency(x: np.ndarray, sample_rate_hz: int) -> float:
    """Estimate frequency from sign changes; appropriate mainly for clean single tones."""
    x = np.asarray(x, dtype=float) - float(np.mean(x))
    signs = x >= 0
    crossings = int(np.count_nonzero(signs[1:] != signs[:-1]))
    duration = len(x) / float(sample_rate_hz)
    return float(crossings / (2.0 * duration)) if duration > 0 else 0.0


def fft_peak_frequency(x: np.ndarray, sample_rate_hz: int) -> float:
    """Return strongest non-DC rFFT bin with Hann windowing."""
    x = np.asarray(x, dtype=float)
    if len(x) < 4:
        return 0.0
    spectrum = np.abs(np.fft.rfft((x - np.mean(x)) * np.hanning(len(x))))
    if len(spectrum) < 2:
        return 0.0
    index = int(np.argmax(spectrum[1:]) + 1)
    return float(index * sample_rate_hz / len(x))


def extract_features(x: np.ndarray, sample_rate_hz: int) -> dict[str, float]:
    return {
        "rms": rms(x),
        "crest_factor": crest_factor(x),
        "kurtosis": kurtosis(x),
        "zero_crossing_frequency_hz": zero_crossing_frequency(x, sample_rate_hz),
        "fft_peak_frequency_hz": fft_peak_frequency(x, sample_rate_hz),
    }
