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


def normalized_band_energies(
    x: np.ndarray,
    sample_rate_hz: int,
    freq_min_hz: float = 0.0,
    freq_max_hz: float = 1200.0,
    num_bands: int = 16,
) -> np.ndarray:
    """Compute normalized energy across fixed frequency bands from rFFT.

    Energy in each band is computed as the sum of squared magnitudes of rFFT bins
    within [band_start, band_end). The resulting band energies are normalized by the
    total energy across all bands (plus an epsilon) so the vector sums to ~1.0.
    """
    x = np.asarray(x, dtype=float)
    if len(x) < 4:
        return np.zeros(num_bands, dtype=float)

    windowed = (x - np.mean(x)) * np.hanning(len(x))
    spectrum = np.abs(np.fft.rfft(windowed))
    power = spectrum**2
    freqs = np.fft.rfftfreq(len(x), 1.0 / sample_rate_hz)

    band_edges = np.linspace(freq_min_hz, freq_max_hz, num_bands + 1)
    band_energies = np.zeros(num_bands, dtype=float)

    for i in range(num_bands):
        f_low = band_edges[i]
        f_high = band_edges[i + 1]
        # Include left edge, include right edge on last band
        if i == num_bands - 1:
            idx = (freqs >= f_low) & (freqs <= f_high)
        else:
            idx = (freqs >= f_low) & (freqs < f_high)
        if np.any(idx):
            band_energies[i] = float(np.sum(power[idx]))

    total_energy = float(np.sum(band_energies))
    if total_energy > 1e-12:
        band_energies = band_energies / total_energy
    else:
        band_energies = np.zeros(num_bands, dtype=float)

    return band_energies


def extract_spectral_features(
    x: np.ndarray,
    sample_rate_hz: int,
    freq_min_hz: float = 0.0,
    freq_max_hz: float = 1200.0,
    num_bands: int = 16,
) -> tuple[np.ndarray, float]:
    """Compute normalized band energies AND peak frequency in a single rFFT pass.

    Eliminates duplicate windowing and rFFT computation while preserving exact
    feature values.
    Returns:
        (band_energies, peak_freq_hz)
    """
    x = np.asarray(x, dtype=float)
    if len(x) < 4:
        return np.zeros(num_bands, dtype=float), 0.0

    n = len(x)
    windowed = (x - np.mean(x)) * np.hanning(n)
    spectrum = np.abs(np.fft.rfft(windowed))

    # Peak frequency: strongest non-DC bin
    if len(spectrum) < 2:
        peak_freq_hz = 0.0
    else:
        peak_idx = int(np.argmax(spectrum[1:]) + 1)
        peak_freq_hz = float(peak_idx * sample_rate_hz / n)

    # 16-band normalized energy accumulation
    power = spectrum**2
    freqs = np.fft.rfftfreq(n, 1.0 / sample_rate_hz)
    band_edges = np.linspace(freq_min_hz, freq_max_hz, num_bands + 1)
    band_energies = np.zeros(num_bands, dtype=float)

    for i in range(num_bands):
        f_low = band_edges[i]
        f_high = band_edges[i + 1]
        if i == num_bands - 1:
            idx = (freqs >= f_low) & (freqs <= f_high)
        else:
            idx = (freqs >= f_low) & (freqs < f_high)
        if np.any(idx):
            band_energies[i] = float(np.sum(power[idx]))

    total_energy = float(np.sum(band_energies))
    if total_energy > 1e-12:
        band_energies = band_energies / total_energy
    else:
        band_energies = np.zeros(num_bands, dtype=float)

    return band_energies, peak_freq_hz

