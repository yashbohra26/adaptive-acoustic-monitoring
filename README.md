# Adaptive Acoustic Signal Processing for Machine Monitoring

> **Project status:** Phase 4 reopened for development-only selector review. A 5-fold grouped cross-validation candidate search has diagnosed the recall failure mode and identified the accuracy-work Pareto trade-off. Held-out test data (20 groups) remains strictly locked. Phase 5, Phase 6, and Phase 7 remain unexecuted/deferred.

## Project in one minute

We are building a software-only study of whether a context-aware algorithm can analyze simulated machine-like sounds with less computational work while retaining useful detection and frequency-estimation quality. Python is the main experimental platform. Wokwi will later demonstrate the selection flow on a simulated ESP32.

We plan to compare three methods on the same labelled signal windows:

1. **Fixed-low:** use inexpensive signal features and a calibrated rule.
2. **Fixed-high:** always use a specified FFT-based analysis and calibrated rule.
3. **Adaptive:** begin with inexpensive features and escalate to FFT when the evidence is uncertain.

The adaptive method must show both its output quality and how often it escalates. Python runtime and estimated operation counts will be reported separately. Neither is a measurement of ESP32 energy use.

## Research question

**Across held-out noise and signal conditions, can an adaptive acoustic-processing policy meet defined detection and frequency-estimation requirements while using less computational work than always running detailed analysis?**

This is a hypothesis to test, not a result we assume in advance. Simulation results apply to the signal conditions we create; they do not by themselves establish performance on real machines or prove novelty or patentability.

## Current build

- [Configuration](python/config.py): sample rate, duration, case ranges, SNRs, split fraction, and reproducible seed.
- [Signal generator](python/generate_signals.py): creates normal, frequency-shift, added-harmonic, and impact windows with known labels, parameters, and grouped development (fit/calibration) / test splits.
- [Feature functions](python/features.py): RMS, crest factor, kurtosis, zero-crossing frequency, FFT peak estimate, and normalized band energies (16 bands, 0-1200 Hz).
- [Baseline detectors](python/baselines.py): Fixed-Low (log time-domain features) and Fixed-High (16-band normalized spectral energies) with robust distance scoring and calibrated thresholds.
- [Adaptive selector](python/adaptive_selector.py): two-stage context-aware detector with deterministic escalation boundaries and complete work accounting.
- [Phase 2 pilot runner](python/run_phase2_pilot.py): generates 80-group pilot dataset, fits reference models on fit groups, and calibrates thresholds on calibration groups.
- [Phase 3 pilot runner](python/run_phase3_pilot.py): derives adaptive boundaries and evaluates development-only escalation distributions by case and SNR.
- [Phase 4 development review](python/run_phase4_review.py): executes development-only diagnostic review across all three methods with group-level bootstrap CIs.
- [Phase 4 grouped CV study](python/run_phase4_study.py): executes 5-fold grouped cross-validation candidate search across 12 selector configurations.
- [Evaluation protocol](docs/EVALUATION_PROTOCOL.md) & [machine-readable protocol](results/evaluation_protocol.json): v1.0 baseline and Section 4 candidate analysis.
- [CV study results](results/phase4_study/selector_study_results.json): out-of-fold metrics and quality constraint evaluation.
- [Phase plan](docs/PROJECT_PLAN.md): deliverables and acceptance checks for each phase.
- [Experiment specification](docs/EXPERIMENT_SPEC.md): shared rules, baseline rules, and adaptive selector policy.
- [AI handoff guide](docs/AI_HANDOFF.md): how to use Antigravity and return its changes for review.
- [AI project instructions](AGENTS.md): concise context and constraints for coding agents.
- [Contributing guide](CONTRIBUTING.md): setup, collaboration, and change-review instructions for teammates.

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
│   ├── run_phase2_pilot.py
│   ├── run_phase3_pilot.py
│   ├── run_phase4_review.py
│   └── run_phase4_study.py
├── results/
│   ├── preview/
│   ├── phase2_pilot/
│   ├── phase3_pilot/
│   ├── phase4_review/
│   ├── phase4_study/
│   │   ├── identifiability_audit.json
│   │   └── selector_study_results.json
│   └── evaluation_protocol.json
├── tests/
│   ├── test_phase1.py
│   ├── test_phase2.py
│   ├── test_phase3.py
│   ├── test_phase4.py
│   ├── test_phase4_cv.py
│   └── test_phase4_investigation.py
└── docs/
    ├── PROJECT_PLAN.md
    ├── EXPERIMENT_SPEC.md
    ├── EVALUATION_PROTOCOL.md
    ├── PHASE4_INVESTIGATION.md
    └── AI_HANDOFF.md
```

Later phases will add method implementations, experiment outputs, and an `esp32_demo/` folder. Do not treat future folders as already implemented.
