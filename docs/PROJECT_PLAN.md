# Project plan and phases

## Scope

Build and evaluate a context-aware selection policy for acoustic signal analysis using software-generated machine-like sounds. Compare fixed-low, fixed-high, and adaptive processing in Python, then demonstrate the selection flow in Wokwi on an ESP32 simulation.

**Not in the current scope:** buying hardware, claiming measured device energy, identifying exact failed machine components, and writing the final report. Report writing can be resumed after the implementation and results are ready.

## Phase status

| Phase | Status | Purpose |
|---|---|---|
| 0. Scope and experiment contract | Done | Set the research question, labels, comparison principles, and evidence limits. |
| 1. Signal and feature foundation | In progress | Generate labelled signals, preserve grouped splits, compute basic features, inspect previews. |
| 2. Detection and frequency baselines | Not started | Define and implement explicit fixed-low and fixed-high rules, calibrated on development data only. |
| 3. Adaptive selector | Not started | Add confidence/uncertainty criteria, escalation stages, and overhead accounting. |
| 4. Pilot and protocol freeze | Not started | Inspect errors, correct design issues, then freeze thresholds, splits, and metrics before final testing. |
| 5. Full evaluation | Not started | Evaluate held-out cases, robustness across noise, sensitivity to selector thresholds, runtime, operation proxy, and escalation rate. |
| 6. Wokwi ESP32 demonstration | Not started | Demonstrate firmware decision flow using deterministic simulated or predefined samples. |
| 7. Research report and presentation | Deferred | Write after implementation and evaluation results are complete. |

## Phase 0 — Scope and experiment contract

**Goal:** everyone on the team shares one project definition.

**Decisions:**

- Acoustic analysis is the selected direction.
- The first output is anomaly detection plus frequency estimation when the signal has a defined target frequency.
- The three methods are fixed-low, fixed-high, and adaptive escalation.
- Python is the research platform; Wokwi is a supporting embedded-style demonstration.
- No physical microphone or ESP32 measurements are claimed.

**Acceptance checks:** research question and limits are documented; signal labels and evaluation rules are explicit enough to implement.

## Phase 1 — Signal and feature foundation (current)

**Build:**

- Reproducible configuration and seeds.
- Cases: normal periodic tone, changed fundamental, added harmonic, short impact.
- Multiple SNR levels for normal and abnormal cases.
- Grouped development/test split by source recipe, keeping noise variants of the same recipe together.
- Feature calculations and waveform/spectrum preview.

**Acceptance checks:**

- Same seed and settings reproduce the same metadata and samples.
- Group IDs do not overlap between development and test.
- Every row has a correct anomaly label and recorded generation settings.
- Target frequency is defined for single-tone normal/shift cases and blank for multi-tone/impact cases.
- Basic feature calculations behave sensibly on known tones and impulses.

**Current remaining work:** inspect generated cases; decide whether the initial signal distribution and group count are adequate; optionally add a reverberation model after the clean/noisy pipeline is stable.

## Phase 2 — Fixed baselines

**Build:**

- Fixed-low detector using inexpensive features only, with a documented decision rule.
- Fixed-high detector using FFT-based spectral evidence and a documented decision rule.
- Development-only calibration for both methods under the same target detection requirement.
- Frequency output for both where the case defines a target.

**Key rule:** RMS alone is signal level, not a reliable definition of abnormality. Calibrate it and evaluate loud-normal false alarms and quieter anomalies. FFT computation alone is not an anomaly detector; specify what spectral evidence determines the label.

**Acceptance checks:** both methods emit anomaly decisions and frequency estimates; thresholds and calibration data are recorded; no test-set tuning occurs.

## Phase 3 — Adaptive selector

**Build:**

- A cheap first stage with an explicit confidence or uncertainty measure.
- Escalation to more detailed statistics and/or FFT only under documented conditions.
- A record of the last stage used for each window.
- Work accounting that includes the selector’s own calculations.

**Acceptance checks:** report detection/frequency quality together with escalation share. If nearly every window reaches FFT, or quality collapses when FFT is avoided, report that honestly and revise only using development data.

## Phase 4 — Pilot and protocol freeze

Run a small pilot to find implementation and methodological issues. After fixing them, freeze:

- Generator ranges and split IDs
- Feature definitions and detector rules
- Thresholds and selector settings
- Evaluation metrics and work model

Record the frozen configuration before opening the final test results.

## Phase 5 — Full evaluation

Run every method on the exact same held-out signal windows. Produce:

- Precision, recall, F1, and false-alarm rate
- Frequency error on valid target cases
- Python runtime per window, reported separately
- Documented operation-count proxy, reported separately
- Adaptive escalation/stage distribution
- Performance versus noise level
- Sensitivity analysis for adaptive thresholds

Do not describe operation counts or desktop runtime as measured ESP32 energy savings.

## Phase 6 — Wokwi ESP32 demonstration

Port a small deterministic version of the decision flow. Print the window, selected stage, and result to Serial. The demo is for firmware-flow illustration; it does not model microphone acoustics or validate physical timing/energy.

## Phase 7 — Report and presentation (deferred)

Only after implementation results are ready, write the methods, results, limitations, and conclusion. Do not invent favorable outcomes: include cases where the adaptive approach loses or fails.

## Team workflow with Antigravity

Work one phase at a time:

1. Ask Antigravity to inspect the repo instructions and the current phase.
2. Give it the phase-specific prompt from `AI_HANDOFF.md`.
3. Ask it to implement only the requested slice and list files changed and commands run.
4. Bring its full response and diff/changed files back here.
5. Review methodology and code together before moving to the next phase.

Do not ask it to implement the entire project in one prompt; this project depends on agreeing and freezing experimental choices before the next stage.
