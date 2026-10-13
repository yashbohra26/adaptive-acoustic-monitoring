# Experiment specification (living document)

This document records rules required for a fair and reproducible method comparison. Decisions marked **OPEN** must be resolved before implementing the corresponding phase.

## Tasks and labels

The current generator labels `normal` as non-anomalous and `frequency_shift`, `added_harmonic`, and `impact` as anomalous. Each signal has a source `group_id`. SNR variants from one group remain in the same split.

Frequency-estimation ground truth is meaningful only for stable single-tone normal and frequency-shift windows. Multi-tone and transient windows currently have no single target frequency; exclude them from frequency-error scoring unless a specific target definition is added first.

## Methods and fairness requirements

### Fixed-low (implemented pilot baseline)

- **Features:** 3 inexpensive time-domain features: \([\log(\text{RMS}), \log(\text{crest factor}), \log(\text{kurtosis})]\).
- **Fitting:** Normal model-fit windows only are used to compute feature-wise medians and robust scales (\(\text{MAD} \times 1.4826\), with scale floor \(10^{-4}\)).
- **Anomaly score:** Standardized Euclidean distance (robust z-score distance):
  \[
  d(\mathbf{x}) = \sqrt{\sum_{i=1}^3 \left(\frac{x_i - \text{median}_i}{\text{scale}_i}\right)^2}
  \]
- **Decision rule & calibration:** Predict anomaly (\(1\)) if \(d(\mathbf{x}) > \tau_{\text{low}}\). Threshold \(\tau_{\text{low}}\) is calibrated as the \((1 - \text{FAR}_{\text{target}})\) percentile (e.g., 95th percentile for 5% target FAR) exclusively on normal threshold-calibration windows. Thresholds remain **provisional** until Phase 4 protocol freeze.
- **Frequency estimation:** Zero-crossing frequency estimate (\(\text{zero\_crossing\_frequency}\)), evaluated only for single-tone cases with defined target frequency.

### Fixed-high (implemented pilot baseline)

- **Features:** 16-dimensional normalized band energy vector computed from Hann-windowed rFFT across 0–1200 Hz (uniform bands of 75 Hz bandwidth, normalized to unit sum).
- **Fitting:** Normal model-fit windows only are used to compute band-wise medians and robust scales (\(\text{MAD} \times 1.4826\), with scale floor \(10^{-4}\)).
- **Anomaly score:** Standardized Euclidean distance (robust z-score distance):
  \[
  d(\mathbf{x}) = \sqrt{\sum_{i=1}^{16} \left(\frac{x_i - \text{median}_i}{\text{scale}_i}\right)^2}
  \]
- **Decision rule & calibration:** Predict anomaly (\(1\)) if \(d(\mathbf{x}) > \tau_{\text{high}}\). Threshold \(\tau_{\text{high}}\) is calibrated as the \((1 - \text{FAR}_{\text{target}})\) percentile (e.g., 95th percentile for 5% target FAR) exclusively on normal threshold-calibration windows. Thresholds remain **provisional** until Phase 4 protocol freeze.
- **Frequency estimation:** Strongest non-DC rFFT peak frequency estimate (\(\text{fft\_peak\_frequency}\)), evaluated only for single-tone cases with defined target frequency.

### Adaptive (unimplemented)

Start with the chosen cheap features and use a documented confidence/uncertainty policy to decide whether to stop or escalate. **OPEN:** exact confidence calculation, stages, and thresholds. Include the selector’s overhead in the work estimate. Unimplemented in Phase 2.

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
