"""Generate labelled, grouped synthetic acoustic windows and preview plots."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np

from config import DEFAULT_CONFIG, SIGNAL_CASES, StudyConfig
from features import extract_features


def synthesize_case(case: str, f0: float, snr_db: float, rng: np.random.Generator,
                    noise_rng: np.random.Generator | None = None,
                    config: StudyConfig = DEFAULT_CONFIG) -> tuple[np.ndarray, float | None]:
    n = config.samples_per_window
    t = np.arange(n, dtype=float) / config.sample_rate_hz
    phase = rng.uniform(0, 2 * np.pi)
    amplitude = rng.uniform(0.65, 1.0)
    if case in ("normal", "frequency_shift"):
        target = f0 if case == "normal" else f0 * rng.uniform(*config.shift_ratio_range)
        clean = amplitude * np.sin(2 * np.pi * target * t + phase)
    elif case == "added_harmonic":
        target = None  # multi-tone: no single frequency-estimation target
        clean = amplitude * (np.sin(2 * np.pi * f0 * t + phase)
                             + 0.45 * np.sin(2 * np.pi * 2 * f0 * t + phase / 2))
    elif case == "impact":
        target = None  # transient carrier is not a stable machine fundamental
        carrier = rng.uniform(*config.impact_carrier_range_hz)
        center = rng.uniform(0.30, 0.70) * config.duration_s
        envelope = np.exp(-np.maximum(t - center, 0) / config.impact_decay_s)
        envelope[t < center] = 0.0
        clean = amplitude * envelope * np.sin(2 * np.pi * carrier * (t - center))
    else:
        raise ValueError(f"Unknown signal case: {case}")
    signal_rms = float(np.sqrt(np.mean(clean**2)))
    noise_rms = signal_rms / (10 ** (snr_db / 20)) if signal_rms > 0 else 0.0
    noise_rng = noise_rng or rng
    noisy = clean + noise_rng.normal(0.0, noise_rms, n)
    return noisy.astype(np.float64), target


def build_dataset(groups: int, config: StudyConfig = DEFAULT_CONFIG):
    """Return records and in-memory signals; split entire source groups together."""
    if groups < 4:
        raise ValueError("Use at least 4 groups to create development and test splits")
    rng = np.random.default_rng(config.seed)
    recipe_rng = np.random.default_rng(config.seed + 1)
    group_specs = [(f"g{i:04d}", float(recipe_rng.uniform(*config.normal_frequency_range_hz)))
                   for i in range(groups)]
    order = rng.permutation(groups)
    n_test = max(1, round(groups * config.test_fraction))
    test_ids = {group_specs[int(i)][0] for i in order[:n_test]}
    records, signals = [], {}
    for group_id, f0 in group_specs:
        split = "test" if group_id in test_ids else "development"
        for case in SIGNAL_CASES:
            for snr in config.snr_db_levels:
                # Keep the clean recipe fixed across SNRs; vary only the noise realization.
                recipe_seed = config.seed + sum(map(ord, group_id + case))
                noise_seed = recipe_seed + int((snr + 50) * 100)
                x, target = synthesize_case(case, f0, snr, np.random.default_rng(recipe_seed),
                                            np.random.default_rng(noise_seed), config)
                key = (group_id, case, snr)
                signals[key] = x
                records.append({
                    "group_id": group_id,
                    "split": split,
                    "case": case,
                    "anomaly": int(case != "normal"),
                    "snr_db": snr,
                    "nominal_normal_frequency_hz": f0,
                    "target_frequency_hz": "" if target is None else target,
                    "n_samples": len(x),
                    "signal_key": key,
                })
    return records, signals


def write_metadata(records, path: Path) -> None:
    fields = [k for k in records[0] if k != "signal_key"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{k: r[k] for k in fields} for r in records])


def save_preview(records, signals, out: Path, config: StudyConfig) -> None:
    examples = []
    for case in SIGNAL_CASES:
        row = next(r for r in records if r["case"] == case and r["snr_db"] == 20)
        examples.append((row, signals[row["signal_key"]]))
    feature_rows = []
    for row, x in examples:
        feature_rows.append({"case": row["case"], "snr_db": row["snr_db"],
                             "target_frequency_hz": row["target_frequency_hz"],
                             **extract_features(x, config.sample_rate_hz)})
    with (out / "feature_preview.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(feature_rows[0]))
        writer.writeheader()
        writer.writerows(feature_rows)
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        # Dependency-free vector chart fallback keeps the first milestone usable offline.
        save_svg_preview(examples, out / "signal_preview.svg", config)
        return
    fig, axes = plt.subplots(len(examples), 2, figsize=(10, 9))
    for i, (row, x) in enumerate(examples):
        time_ms = np.arange(len(x)) / config.sample_rate_hz * 1000
        freq = np.fft.rfftfreq(len(x), 1 / config.sample_rate_hz)
        mag = np.abs(np.fft.rfft((x - np.mean(x)) * np.hanning(len(x))))
        axes[i, 0].plot(time_ms, x, linewidth=0.8)
        axes[i, 0].set(title=f"{row['case']} waveform (SNR {row['snr_db']:.0f} dB)", xlabel="Time (ms)", ylabel="Amplitude")
        axes[i, 1].plot(freq, mag, linewidth=0.8)
        axes[i, 1].set(xlim=(0, 1200), title="Magnitude spectrum", xlabel="Frequency (Hz)", ylabel="Magnitude")
    fig.tight_layout()
    fig.savefig(out / "signal_preview.png", dpi=160)
    plt.close(fig)


def save_svg_preview(examples, path: Path, config: StudyConfig) -> None:
    """Write waveform/spectrum panels as SVG using only the Python standard library."""
    width, row_h, left_w, right_w = 1000, 230, 470, 470
    height = row_h * len(examples)
    chunks = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
              '<rect width="100%" height="100%" fill="#ffffff"/>',
              '<style>text{font-family:Arial,sans-serif;fill:#243247} .grid{stroke:#d9e1eb;stroke-width:1} .wave{fill:none;stroke:#2878b5;stroke-width:1.4} .spec{fill:none;stroke:#d06b35;stroke-width:1.4}</style>']
    for i, (row, signal) in enumerate(examples):
        top = i * row_h
        chunks.append(f'<text x="20" y="{top+23}" font-size="17" font-weight="bold">{row["case"]} · SNR {row["snr_db"]:.0f} dB</text>')
        for x0, panel_w, title in ((35, left_w, "Waveform (time in ms)"), (520, right_w, "Magnitude spectrum (Hz)")):
            chunks.append(f'<text x="{x0}" y="{top+49}" font-size="13">{title}</text>')
            chunks.append(f'<line class="grid" x1="{x0}" y1="{top+160}" x2="{x0+panel_w}" y2="{top+160}"/>')
        x0, y0, plot_w, plot_h = 35, top + 65, left_w, 145
        max_abs = max(float(np.max(np.abs(signal))), 1e-9)
        stride = max(1, len(signal) // plot_w)
        pts = []
        for j in range(0, len(signal), stride):
            px = x0 + j / (len(signal)-1) * plot_w
            py = y0 + plot_h / 2 - signal[j] / max_abs * (plot_h / 2 - 3)
            pts.append(f'{px:.1f},{py:.1f}')
        chunks.append(f'<polyline class="wave" points="{" ".join(pts)}"/>')
        spectrum = np.abs(np.fft.rfft((signal - np.mean(signal)) * np.hanning(len(signal))))
        freqs = np.fft.rfftfreq(len(signal), 1 / config.sample_rate_hz)
        mask = freqs <= 1200
        max_mag = max(float(np.max(spectrum[mask])), 1e-9)
        pts = []
        for f, mag in zip(freqs[mask], spectrum[mask]):
            px = 520 + f / 1200 * right_w
            py = y0 + plot_h - mag / max_mag * (plot_h - 3)
            pts.append(f'{px:.1f},{py:.1f}')
        chunks.append(f'<polyline class="spec" points="{" ".join(pts)}"/>')
    chunks.append('</svg>')
    path.write_text('\n'.join(chunks), encoding='utf-8')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("results/preview"))
    parser.add_argument("--groups", type=int, default=12)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    records, signals = build_dataset(args.groups)
    for split in ("development", "test"):
        write_metadata([r for r in records if r["split"] == split], args.out / f"{split}_metadata.csv")
    save_preview(records, signals, args.out, DEFAULT_CONFIG)
    print(f"Generated {len(records)} windows across {args.groups} source groups in {args.out}")
    print("Files: development_metadata.csv, test_metadata.csv, signal_preview.svg or .png, feature_preview.csv")


if __name__ == "__main__":
    main()
