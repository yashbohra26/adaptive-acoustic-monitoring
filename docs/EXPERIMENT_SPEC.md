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

### Adaptive (implemented pilot selector)

- **Two-stage policy:**
  1. Evaluate the Fixed-Low detector first to obtain low-stage standardized distance score \(d_{\text{low}}(\mathbf{x})\) and zero-crossing frequency estimate \(f_{\text{zcf}}(\mathbf{x})\).
  2. Boundaries derived **exclusively from normal threshold-calibration windows**:
     - Low score 90th percentile: \(\tau_{90}\) (e.g., \(\approx 5.0044\))
     - Low score 99th percentile: \(\tau_{99}\) (e.g., \(\approx 5.5732\))
     - Zero-crossing frequency 1st and 99th percentiles: \(f_{\text{zcf}, 1\%}\) and \(f_{\text{zcf}, 99\%}\) (e.g., \(\approx 183.96\) Hz and \(644.63\) Hz)
  3. **Decision & Escalation Logic:**
     - If \(f_{\text{zcf}}(\mathbf{x}) < f_{\text{zcf}, 1\%}\) or \(f_{\text{zcf}}(\mathbf{x}) > f_{\text{zcf}, 99\%}\): Escalate to Stage 2 (Fixed-High FFT) due to out-of-range zero-crossing frequency (`zcf_out_of_range`).
     - Else if \(d_{\text{low}}(\mathbf{x}) \le \tau_{90}\): Stop at Stage 1 as Confidently Normal (`confident_normal`, label 0, `fft_used=False`).
     - Else if \(d_{\text{low}}(\mathbf{x}) \ge \tau_{99}\): Stop at Stage 1 as Confidently Anomalous (`confident_anomaly`, label 1, `fft_used=False`).
     - Else (\(\tau_{90} < d_{\text{low}}(\mathbf{x}) < \tau_{99}\)): Escalate to Stage 2 (Fixed-High FFT) due to ambiguous score (`uncertain_score`).
  4. If escalated, the window is processed by Fixed-High, and its calibrated prediction and FFT peak frequency estimate are adopted.
- **Work accounting (Operation-Count Proxy):**
  - `stage_reached`: `"stage1_low"` or `"stage2_high"`.
  - `fft_used`: Boolean indicator recorded for every window.
  - `op_count_proxy`: Operation-count proxy strictly matching executed stage operations:
    - Stage 1 alone: \(10 N + 25\) ops (\(10,265\) ops for \(N=1024\)).
    - Fixed-High standalone: \(2.5 N \log_2(N) + 3.5 N + 66\) ops (\(29,250\) ops for \(N=1024\)).
    - Stage 2 cumulative (escalated window): Stage 1 + Stage 2 = \(10,265 + 29,250 = 39,515\) ops.
    - Adaptive weighted mean: \((N_{\text{stage1}} \times 10,265 + N_{\text{stage2}} \times 39,515) / N_{\text{total}}\).
  - Single-pass FFT implementation (`extract_spectral_features` in `python/features.py`) unifies 16-band normalized energy and FFT peak frequency into a single rFFT call, eliminating duplicate FFT execution and strictly aligning code execution with the 29,250 ops proxy model.
  - Desktop execution time measured under uniform benchmark conditions (5 warmup iterations, 25 timing repetitions per window using `time.perf_counter_ns()`). Reported separately from operation counts. Neither represents physical hardware energy.
- **Protocol Versions:**
  - **Protocol v1.0.0 (Historical):** Frozen in [`results/evaluation_protocol.json`](../results/evaluation_protocol.json).
  - **Protocol v1.1.0 (Phase 5 Reference):** Frozen in [`results/evaluation_protocol_v1.1.json`](../results/evaluation_protocol_v1.1.json). Evaluated on original 20 held-out test groups (320 windows, SHA-256 `b99e0b36...`). Fixed-High: F1 0.9492, Recall 0.9333, FAR 0.1000, 29,250 ops. Candidate v1.1: F1 0.9170, Recall 0.8750 (recall constraint $< 0.8833$ not met due to subtle harmonic misses), FAR 0.1000, 26,992.3 ops/window (+7.72% operational savings). Documented in [`results/phase5_final_evaluation.json`](../results/phase5_final_evaluation.json).
  - **Protocol v1.2.0 (Harmonic-Sensitive Candidate):** Frozen in [`results/evaluation_protocol_v1.2.json`](../results/evaluation_protocol_v1.2.json) following development-only 5-fold cross-validation ($P_{\text{normal}} = \text{p50} = 1.3110$). Evaluated on a fresh independent held-out partition (20 new groups, seed=20261014; legacy group hash `5b918927...`, canonical content hash `96264358...`): observed Fixed-High F1 0.9534 [0.9231, 0.9793], Recall 0.9375 [0.8792, 0.9833]; Candidate v1.2 observed F1 0.9534 [0.9231, 0.9793], Recall 0.9375 [0.8792, 0.9833], FAR 0.0875 [0.0375, 0.1375], 28,729.1 ops/window (+1.78% observed savings; 95% bootstrap interval [-1.66%, +4.91%], which spans zero). Full provenance and 1,000-iteration group bootstrap 95% CIs recorded in [`results/candidate_v1.2_provenance_and_uncertainty.json`](../results/candidate_v1.2_provenance_and_uncertainty.json); figures and tables in `results/figures/` and `results/tables/`.
- **Phase 6 Embedded Demonstration:** Implemented under `esp32_demo/` for Wokwi ESP32 emulation, porting the frozen decision hierarchy. Default firmware (`src/main.cpp`) implements Candidate v1.1 (`cand_v1.1_p60_anomaly_first`). Candidate v1.2 is currently provided and verified in a separate desktop C++ core/harness ([`esp32_demo/src/adaptive_core_v1_2.h`](../esp32_demo/src/adaptive_core_v1_2.h), [`esp32_demo/test_harness.cpp`](../esp32_demo/test_harness.cpp), 100% full parity) and is not called a Wokwi demo. Live Wokwi interactive simulation remains pending manual launch until serial monitor output is directly confirmed.

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
