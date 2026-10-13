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
  - `op_count_proxy`: Operation-count proxy including selector overhead:
    - Stage 1 alone: \(10 N + 25\) ops (\(10,265\) ops for \(N=1024\)).
    - Fixed-High standalone: \(2.5 N \log_2(N) + 3.5 N + 66\) ops (\(29,250\) ops for \(N=1024\)).
    - Stage 2 cumulative (escalated window): Stage 1 + Stage 2 = \(10,265 + 29,250 = 39,515\) ops.
    - Adaptive weighted mean: \((N_{\text{stage1}} \times 10,265 + N_{\text{stage2}} \times 39,515) / N_{\text{total}}\).
    - For development pilot: \((606 \times 10,265 + 354 \times 39,515) / 960 = 21,050.9\) ops/window (\(-28.0\%\) relative to standalone Fixed-High).
  - Desktop execution time in milliseconds recorded separately from operation counts. Neither represents physical hardware energy.
- **Protocol Freeze:** Historical v1.0 specified and frozen in [`docs/EVALUATION_PROTOCOL.md`](EVALUATION_PROTOCOL.md) and [`results/evaluation_protocol.json`](../results/evaluation_protocol.json). Active Protocol v1.1.0 frozen in [`results/evaluation_protocol_v1.1.json`](../results/evaluation_protocol_v1.1.json).
- **Candidate Selector v1.1 Investigation:** Fully analyzed in [`docs/PHASE4_INVESTIGATION.md`](PHASE4_INVESTIGATION.md) and evaluated via 5-fold CV (`cand_v1.1_p60_anomaly_first` in `python/candidate_selector.py`).
- **Phase 5 Held-Out Evaluation:** Completed on the 20 held-out test source groups (320 windows). Fixed-High F1 0.9492 (29,250 ops); Candidate v1.1 F1 0.9170, Recall 0.8750, FAR 0.1000, 26,992.3 ops/window (+7.72% operational savings). Documented in [`results/phase5_final_evaluation.json`](../results/phase5_final_evaluation.json).
- **Phase 6 Embedded Demonstration:** Implemented under `esp32_demo/` for Wokwi ESP32 emulation, porting the frozen Protocol v1.1.0 decision hierarchy. Verified 100% decision and stage parity on 5 deterministic development vectors from Group `g0001` ([`esp32_demo/README.md`](../esp32_demo/README.md)).

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
