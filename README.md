# Adaptive Acoustic Signal Processing for Machine Monitoring

> **Project status:** Phase 5 held-out evaluation and Phase 6 embedded firmware port completed. Redundant FFT computation resolved; operation cost proxy audited and aligned. Fair three-method benchmarking completed with repeated warm desktop timing. Automated quality constraint checks enforce zero duplicate FFTs and transparent constraint reporting. Development-only investigation identified Candidate v1.2, frozen under Protocol v1.2.0 and validated on an independent holdout. Wokwi live simulation remains pending manual execution. Phase 7 research report remains deferred.

## Project in one minute

We are building a software-only study of whether a context-aware algorithm can analyze simulated machine-like sounds with less computational work while retaining useful detection and frequency-estimation quality. Python is the main experimental platform. Wokwi will demonstrate the selection flow on a simulated ESP32.

We compare three methods on the same labelled signal windows:

1. **Fixed-low:** uses inexpensive time-domain features (log RMS, log crest factor, log kurtosis) and zero-crossing frequency ($10,265$ ops proxy).
2. **Fixed-high:** always uses Hann-windowed single-pass rFFT spectral analysis and 16-band normalized energy ($29,250$ ops proxy).
3. **Adaptive:** begins with inexpensive time-domain features and escalates to spectral FFT when evidence is ambiguous or indicates frequency deviation ($10,265$ ops if Stage 1; $39,515$ ops if escalated).

The adaptive method must demonstrate both its output quality and its computational efficiency. Python desktop runtime and algorithmic operation-count proxy are reported separately. Neither represents physical ESP32 energy consumption.

## Research question

**Across held-out noise and signal conditions, can an adaptive acoustic-processing policy meet defined detection and frequency-estimation requirements while using less computational work than always running detailed analysis?**

This is a hypothesis to test, not a result we assume in advance. Simulation results apply to the signal conditions we create; they do not by themselves establish performance on real machines or prove novelty or patentability.

## Current build

- [Configuration](python/config.py): sample rate (4000 Hz), window duration (0.256 s, 1024 samples), case ranges, SNRs (30, 20, 10, 0 dB), and reproducible seeds.
- [Signal generator](python/generate_signals.py): creates normal, frequency-shift, added-harmonic, and impact windows with known labels, parameters, and grouped development (fit/calibration) / test splits.
- [Feature functions](python/features.py): RMS, crest factor, kurtosis, zero-crossing frequency, FFT peak estimate, normalized band energies, and unified single-pass `extract_spectral_features`.
- [Baseline detectors](python/baselines.py): Fixed-Low and Fixed-High baseline detectors with single-pass FFT execution and calibrated thresholds.
- [Adaptive selector](python/adaptive_selector.py): historical Protocol v1.0 two-stage detector.
- [Candidate selector](python/candidate_selector.py): Candidate v1.1 and v1.2 two-stage selector implementing the anomaly-first decision cascade.
- [Fair benchmarking harness](python/benchmark_methods.py): uniform benchmark across all methods with 5 warmup runs and 25 timing repetitions per window.
- [Difficult-case investigation](python/investigate_difficult_cases.py): development-only CV study, Protocol v1.2.0 freeze, and independent holdout evaluation.
- [Plot & table generator](python/plot_evaluation_results.py): generates publication figures (PNG/PDF) and summary tables (CSV/Markdown).
- [Phase 2–5 evaluation runners](python/): reproducible pipelines for pilots, development review, CV studies, and final test evaluation.
- [ESP32 demonstration](esp32_demo/): pure C++ embedded decision flow with 100% desktop parity; Wokwi platform configuration and demo vectors.
- [Literature review](docs/PRIOR_ART.md): formal prior art review covering Viola-Jones cascades, BranchyNet early-exits, and Randall-Antoni machinery diagnostics.
- [Automated quality constraints](tests/test_quality_constraints.py): 39 comprehensive unit tests verifying zero duplicate FFTs, cost accounting, split disjointness, and constraint reporting.

## Run the current signal-generation milestone

From this project folder:

```powershell
py -m pip install -r requirements.txt
py python/generate_signals.py --out results/preview --groups 8
py -m pytest tests/ -v
```

The generator writes development and test metadata, feature values, and a waveform/spectrum preview. If Python is installed through a different launcher on your computer, use that interpreter to run the same script.

## Important experiment rules

- Keep variants from the same source recipe together in one split; never put close variants in both development and test sets.
- Calibrate all thresholds and selector rules with development data only. Keep the test set untouched until the protocol is frozen.
- Define a real anomaly decision rule for each baseline. Computing an FFT by itself is not a detector.
- Compare all methods on the same held-out windows and report detection quality, frequency error where defined, runtime, operation-count proxy, and adaptive escalation rate.
- `target_frequency_hz` is blank for multi-tone and impact cases because a single correct frequency target is not defined for those signals.
- Keep operation counts, Python runtime, firmware runtime, and energy measurements distinct. We do not have physical hardware or measured energy in this project.

## Folder map

```text
adaptive-acoustic-monitoring/
├── .gitignore
├── AGENTS.md
├── CONTRIBUTING.md
├── README.md
├── requirements.txt
├── python/
│   ├── config.py
│   ├── generate_signals.py
│   ├── features.py
│   ├── candidate_features.py
│   ├── baselines.py
│   ├── adaptive_selector.py
│   ├── candidate_selector.py
│   ├── benchmark_methods.py
│   ├── investigate_difficult_cases.py
│   ├── plot_evaluation_results.py
│   ├── run_phase2_pilot.py
│   ├── run_phase3_pilot.py
│   ├── run_phase4_review.py
│   ├── run_phase4_study.py
│   ├── run_phase5_evaluation.py
│   └── export_demo_vectors.py
├── esp32_demo/
│   ├── platformio.ini
│   ├── wokwi.toml
│   ├── diagram.json
│   ├── test_harness.cpp
│   ├── demo_vectors_metadata.json
│   ├── README.md
│   └── src/
│       ├── main.cpp
│       ├── adaptive_core.h
│       └── demo_vectors.h
├── results/
│   ├── figures/
│   │   ├── fig1_overall_performance.png / .pdf
│   │   ├── fig2_work_and_runtime.png / .pdf
│   │   ├── fig3_case_and_snr_recall.png / .pdf
│   │   ├── fig4_adaptive_escalation.png / .pdf
│   │   └── fig5_error_breakdown.png / .pdf
│   ├── tables/
│   │   ├── overall_performance_table.md / .csv
│   │   └── case_breakdown_table.md / .csv
│   ├── preview/
│   ├── phase2_pilot/
│   ├── phase3_pilot/
│   ├── phase4_review/
│   ├── phase4_study/
│   ├── evaluation_protocol.json
│   ├── evaluation_protocol_v1.1.json
│   ├── evaluation_protocol_v1.2.json
│   ├── phase5_final_evaluation.json
│   ├── phase5_window_predictions.csv
│   ├── benchmark_fair_evaluation.json
│   ├── benchmark_fair_windows.csv
│   ├── candidate_v1.2_development_study.json
│   └── candidate_v1.2_independent_evaluation.json
├── tests/
│   ├── test_phase1.py
│   ├── test_phase2.py
│   ├── test_phase3.py
│   ├── test_phase4.py
│   ├── test_phase4_cv.py
│   ├── test_phase4_investigation.py
│   ├── test_phase5.py
│   ├── test_phase6_wokwi.py
│   └── test_quality_constraints.py
└── docs/
    ├── PROJECT_PLAN.md
    ├── EXPERIMENT_SPEC.md
    ├── EVALUATION_PROTOCOL.md
    ├── PHASE4_INVESTIGATION.md
    ├── PRIOR_ART.md
    └── AI_HANDOFF.md
```

### Phase Status Overview

- **Phase 1 (Signal Foundation):** Complete (`476ae37`).
- **Phase 2 (Fixed Baselines):** Complete (`17bece2`).
- **Phase 3 (Adaptive Selector v1.0):** Complete (`fe9b757`).
- **Phase 4 (Audit & Protocol v1.1.0 Freeze):** Complete (`0dc2b39`, `97614a2`, `02bf90f`, `3e242ee`).
- **Phase 5 (Held-Out Test Evaluation):** Complete (`1a283ae`). Candidate v1.1 achieved F1 = 0.9170, Recall = 0.8750 (recall constraint $< 0.8833$ not met on subtle harmonics), Ops = 26,992.3 (+7.72% savings).
- **Phase 6 (Embedded Firmware Port):** Complete (`ac46a12`, `7856ac8`). C++ desktop harness passes with 100% parity; Wokwi live simulation execution remains pending manual launch.
- **FFT Refactoring & Fair Benchmark:** Single-pass spectral extraction implemented, eliminating duplicate FFT execution. Benchmarked with 5 warmup runs and 25 timing repetitions per window.
- **Candidate v1.2 Development Study & Independent Holdout:** Development CV selected $P_{\text{normal}} = \text{p50}$, frozen under Protocol v1.2.0, resolving harmonic misses and matching Fixed-High quality on an independent holdout (seed 20261014).
- **Phase 7 (Research Report):** Deferred per project instructions.

