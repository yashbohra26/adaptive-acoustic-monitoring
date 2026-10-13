# Project plan and phases

## Scope

Build and evaluate a context-aware selection policy for acoustic signal analysis using software-generated machine-like sounds. Compare fixed-low, fixed-high, and adaptive processing in Python, then demonstrate the selection flow in Wokwi on an ESP32 simulation.

**Not in the current scope:** buying hardware, claiming measured device energy, identifying exact failed machine components, and writing the final report. Report writing can be resumed after the implementation and results are ready.

## Phase status

| Phase | Status | Purpose |
|---|---|---|
| 0. Scope and experiment contract | Done | Set the research question, labels, comparison principles, and evidence limits. |
| 1. Signal and feature foundation | Done | Generate labelled signals, preserve grouped splits, compute basic features, inspect previews. |
| 2. Detection and frequency baselines | Done (Pilot) | Define and implement explicit fixed-low and fixed-high rules, calibrated on development data only. |
| 3. Adaptive selector | Done (Pilot) | Add confidence/uncertainty criteria, escalation stages, and overhead accounting. |
| 4. Pilot and protocol freeze | Review in progress | Development CV review completed; failure modes diagnosed; open for team trade-off decision. |
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

## Phase 2 — Fixed baselines (Pilot Implemented)

**Build:**

- Fixed-low detector using inexpensive features only (log RMS, log crest factor, log kurtosis) with standardized distance from normal fit median/scale.
- Fixed-high detector using FFT-based normalized band energies (16 bands, 0-1200 Hz) with standardized distance from normal fit median/scale.
- Development-only calibration for both methods targeting 5% false-alarm rate on normal threshold-calibration windows.
- Frequency output for both where the case defines a target (zero-crossing for fixed-low, FFT peak for fixed-high).
- 80-group pilot dataset generated (60 development groups split 30/30 into fit/calibration, 20 test groups held out untouched).

**Key rule:** RMS alone is signal level, not a reliable definition of abnormality. Calibrate it and evaluate loud-normal false alarms and quieter anomalies. FFT computation alone is not an anomaly detector; specify what spectral evidence determines the label.

**Acceptance checks:** both methods emit anomaly decisions and frequency estimates; thresholds and calibration data are recorded; no test-set tuning occurs. Fitted parameters and provisional thresholds recorded in `results/phase2_pilot/pilot_calibration.json`.

## Phase 3 — Adaptive selector (Pilot Implemented)

**Build:**

- Two-stage adaptive selector (`AdaptiveSelector`) running `FixedLowDetector` first and conditionally escalating to `FixedHighDetector`.
- Transparent decision boundaries derived strictly from normal calibration data:
  - Low-stage score 90th and 99th percentiles.
  - Zero-crossing frequency 1st and 99th percentiles.
- Deterministic escalation triggers on out-of-range frequency or ambiguous low score.
- Full work accounting: records stage reached (`stage1_low` or `stage2_high`), `fft_used` boolean, execution time, and operation-count proxy including selector overhead.
- Development-only pilot evaluation (`results/phase3_pilot/`) recording escalation distributions by case and SNR without touching test data.

**Acceptance checks:** report detection/frequency quality together with escalation share. Confident cases stop cheaply at stage 1; impacts and out-of-range frequencies escalate reliably; work and runtime recorded separately. Fitted boundaries recorded in `results/phase3_pilot/adaptive_diagnostics.json`.

## Phase 4 — Pilot review, protocol freeze, and candidate feature investigation (Done)

**Completed:**

- Audited operation-count proxy arithmetic: resolved cumulative vs additional stage accounting.
  - Stage 1: \(10,265\) ops; Stage 2 additional: \(29,250\) ops; Escalated window cumulative: \(39,515\) ops.
  - Exact weighted mean verified: \((606 \times 10,265 + 354 \times 39,515) / 960 = 21,050.9375\) ops/window (\(-28.0\%\) vs Fixed-High).
- Initial protocol frozen in versioned human-readable document [`docs/EVALUATION_PROTOCOL.md`](EVALUATION_PROTOCOL.md) and machine-readable specification [`results/evaluation_protocol.json`](../results/evaluation_protocol.json).
- Test split protected with strict programmatic guard and hash verification (`b99e0b36ea5b6659`).
- **Feature Investigation & Identifiability Audit (`docs/PHASE4_INVESTIGATION.md`):**
  - Audited inference-time inputs: verified no leakage of machine identity, nominal \(f_0\), or SNR.
  - Proved theoretical break-even cascade escalation limit: \(E_{\max} = (29,250 - 10,265) / 29,250 = 64.91\%\).
  - Evaluated candidate lightweight features (normalized difference ratio \(R_{\text{diff}}\), ZCF bounds) and analyzed noise smearing at 0 dB SNR.
  - Developed and verified Candidate Selector v1.1 (`cand_v1.1_p60_anomaly_first` in `python/candidate_selector.py`):
    - Re-architected cascade order: confident anomaly checked *before* frequency discrepancy, eliminating unnecessary FFT escalation on impacts (0.0% escalation).
    - Achieved **F1 0.9544** (matches Fixed-High), **Recall 0.9292** (matches Fixed-High), **FAR 0.0542** (matches Fixed-High), **Average Ops 28,211.1** ops/window (**+3.55% positive savings**).
  - Preserved v1.0 frozen protocol as historical baseline while preparing candidate v1.1 for user review.

**Acceptance checks:** all 28 Phase 1–4 tests pass; operation-cost arithmetic is consistent; held-out test data remains uninspected and protected from evaluation.

## Phase 5 — Full evaluation (Completed)

Executed single frozen evaluation on the 20 held-out test source groups (320 windows, SHA-256: `b99e0b36ea5b6659`) under Protocol v1.1.0 (`results/evaluation_protocol_v1.1.json`).

**Final Held-Out Results Summary (with group-level bootstrap 95% CIs):**
- **Fixed-Low Standalone:** F1 = `0.5572` (`[0.5313, 0.5788]`), Recall = `0.3958` (`[0.3750, 0.4167]`), FAR = `0.0750` (`[0.0250, 0.1375]`), Freq MAE = `91.44` Hz, Ops = `10,265.0`.
- **Fixed-High Standalone:** F1 = `0.9492` (`[0.9123, 0.9772]`), Recall = `0.9333` (`[0.8667, 0.9833]`), FAR = `0.1000` (`[0.0500, 0.1500]`), Freq MAE = `0.85` Hz, Ops = `29,250.0`.
- **Adaptive Selector v1.0 (Historical):** F1 = `0.6243` (`[0.6023, 0.6431]`), Recall = `0.4708` (`[0.4458, 0.4917]`), FAR = `0.1125` (`[0.0625, 0.1625]`), Freq MAE = `38.37` Hz, Escalation = `38.12%`, Ops = `21,416.6` (+26.78% savings, but misses 53% of anomalies).
- **Candidate Adaptive Selector v1.1:** F1 = `0.9170` (`[0.8662, 0.9575]`), Recall = `0.8750` (`[0.7957, 0.9458]`), FAR = `0.1000` (`[0.0500, 0.1500]`), Freq MAE = `2.75` Hz, Escalation = `57.19%`, Average Ops = `26,992.3` (`[25,347.0, 28,363.4]`), **+7.72% operational savings**.
  - Matches Fixed-High False-Alarm Rate exactly (0.1000 vs 0.1000).
  - 100% recall on impacts with 0.0% escalation (decided at Stage 1).
  - Matches Fixed-High on frequency shifts (80.00% recall).
  - Misses 14 quiet added-harmonic windows at high SNR that score below Stage 1 normal threshold.

Full outputs: [`results/phase5_final_evaluation.json`](../results/phase5_final_evaluation.json) and [`results/phase5_window_predictions.csv`](../results/phase5_window_predictions.csv).

## Phase 6 — Wokwi ESP32 demonstration (Deferred)

Port a small deterministic version of the decision flow. Print the window, selected stage, and result to Serial. The demo is for firmware-flow illustration; it does not model microphone acoustics or validate physical timing/energy.

## Phase 7 — Report and presentation (Deferred)

Deferred until explicitly requested by user. Do not invent favorable outcomes: include cases where the adaptive approach loses or fails.

## Team workflow with Antigravity

Work one phase at a time:

1. Ask Antigravity to inspect the repo instructions and the current phase.
2. Give it the phase-specific prompt from `AI_HANDOFF.md`.
3. Ask it to implement only the requested slice and list files changed and commands run.
4. Bring its full response and diff/changed files back here.
5. Review methodology and code together before moving to the next phase.

Do not ask it to implement the entire project in one prompt; this project depends on agreeing and freezing experimental choices before the next stage.
