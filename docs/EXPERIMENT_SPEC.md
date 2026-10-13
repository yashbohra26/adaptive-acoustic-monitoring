# Experiment specification (living document)

This document records rules required for a fair and reproducible method comparison. Decisions marked **OPEN** must be resolved before implementing the corresponding phase.

## Tasks and labels

The current generator labels `normal` as non-anomalous and `frequency_shift`, `added_harmonic`, and `impact` as anomalous. Each signal has a source `group_id`. SNR variants from one group remain in the same split.

Frequency-estimation ground truth is meaningful only for stable single-tone normal and frequency-shift windows. Multi-tone and transient windows currently have no single target frequency; exclude them from frequency-error scoring unless a specific target definition is added first.

## Methods and fairness requirements

### Fixed-low

Use inexpensive features only. **OPEN:** choose the exact anomaly score and calibrated rule. RMS by itself is only signal level; it can confuse a loud normal case with an anomaly. Candidate cheap features include RMS, crest factor, and kurtosis.

### Fixed-high

Always compute FFT-based features. **OPEN:** define a spectral anomaly score (for example, deviation from a development-set normal spectral profile) and its calibration rule. An FFT transform alone does not make a prediction.

### Adaptive

Start with the chosen cheap features and use a documented confidence/uncertainty policy to decide whether to stop or escalate. **OPEN:** exact confidence calculation, stages, and thresholds. Include the selector’s overhead in the work estimate.

### Shared requirements

- Use common development data and identical held-out windows.
- Set thresholds/selector using development data only.
- Group related signal recipes before splitting; do not split near-identical noise variants across development and test.
- Record predicted label, frequency estimate, stage reached, runtime, and work estimate for each method/window.
- Never select a rule after inspecting final test outcomes and then report those same outcomes as held out.

## Required measures

- Detection: precision, recall, F1, false-alarm rate (also identify the denominator used).
- Frequency: absolute error in Hz on the explicitly eligible single-tone cases.
- Runtime: measured Python time per window on the same machine/software environment, clearly labelled desktop runtime.
- Work: operation-count proxy with calculation described. Report separately from runtime.
- Adaptation: percentage of windows ending at each stage, including FFT escalation rate.
- Robustness: detection and frequency performance by SNR.
- Sensitivity: performance and escalation under alternative selector thresholds.

## Simulation claims and limits

The current signal generator uses controlled waveform recipes plus additive noise. It does not reproduce all real machine sound, microphone, room, sensor, or factory effects. Reverberation is an optional later extension and must be described with its simulator assumptions. Wokwi can demonstrate control flow but is not an acoustic propagation or power simulator.

Allowed claim style: “On these held-out synthetic conditions, method A achieved …”

Avoid claiming real-machine validity, measured ESP32 energy savings, guaranteed publication, or patentability without separate supporting evidence.
