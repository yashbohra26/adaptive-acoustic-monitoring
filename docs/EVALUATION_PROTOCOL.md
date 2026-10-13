# Frozen Evaluation Protocol (v1.0.0)

> **Status:** FROZEN  
> **Protocol Version:** 1.0.0  
> **Date Frozen:** 2026-10-14  
> **Commit Hash:** Frozen in Phase 4 commit (see repository log)  
> **Machine-Readable Spec:** [`results/evaluation_protocol.json`](../results/evaluation_protocol.json)

---

## 1. Experimental Design and Data Partitions

### 1.1 Dataset Generation Settings
- **Master Random Seed:** `20261013`
- **Sample Rate:** `4000` Hz
- **Window Duration:** `0.256` s (\(N = 1024\) samples per window)
- **SNR Levels:** `[30.0, 20.0, 10.0, 0.0]` dB
- **Frequency Ranges:**
  - Normal nominal fundamental: \(180.0\)–\(260.0\) Hz
  - Frequency shift ratio range: \(1.35\)–\(1.60\)
  - Impact transient carrier range: \(500.0\)–\(900.0\) Hz; decay constant: \(0.009\) s
- **Signal Cases:** `normal`, `frequency_shift`, `added_harmonic`, `impact`

### 1.2 Group-Based Split Policy
To eliminate variant leakage across splits, all 16 variants of a source group (4 cases \(\times\) 4 SNRs) remain strictly clustered in the same split:
- **Total Source Groups:** 80 groups (\(1280\) windows)
- **Held-Out Test Split:** 20 source groups (\(320\) windows). Verifiable SHA256 prefix: `b99e0b36ea5b6659`.
  > **STRICT LOCK:** Test group data and labels are sealed and protected from inspection, feature exploration, model fitting, and threshold tuning throughout Phases 1–4.
- **Development Split:** 60 source groups (\(960\) windows). Verifiable SHA256 prefix: `9ece130e3b4054a7`.
  - **Model-Fit Partition:** 30 groups (\(480\) windows; \(120\) normal windows) used solely for estimating reference medians and MAD scales.
  - **Threshold-Calibration Partition:** 30 groups (\(480\) windows; \(120\) normal windows) used solely for threshold calibration and selector boundary derivation.

---

## 2. Frozen Methods and Decision Rules

### 2.1 Fixed-Low Baseline
- **Features:** \([\log(\text{RMS}), \log(\text{crest factor}), \log(\text{kurtosis})]\) with numerical floor \(10^{-8}\).
- **Reference Parameters (fitted on model-fit normals):**
  - Median: `[-0.4221, 0.6272, 0.4827]`
  - Scale (\(1.4826 \times \text{MAD}\), floor \(10^{-4}\)): `[0.1325, 0.3221, 0.1135]`
- **Anomaly Score:** Standardized Euclidean distance (robust z-score distance):
  \[
  d_{\text{low}}(\mathbf{x}) = \sqrt{\sum_{i=1}^3 \left(\frac{x_i - \text{median}_i}{\text{scale}_i}\right)^2}
  \]
- **Frozen Threshold:** \(\tau_{\text{low}} = 5.274212\) (calibrated to 5% FAR on calibration normals).
- **Decision Rule:** Anomaly (\(1\)) if \(d_{\text{low}}(\mathbf{x}) > \tau_{\text{low}}\); else Normal (\(0\)).
- **Frequency Estimator:** Zero-crossing frequency (\(\text{zero\_crossing\_frequency}\)).

### 2.2 Fixed-High Baseline
- **Features:** 16-band normalized spectral energies from Hann-windowed rFFT across 0–1200 Hz (75 Hz bandwidth per band, normalized to sum to 1.0).
- **Reference Parameters:** Fitted on model-fit normals (16-dimensional median and robust scale vectors; recorded in `results/evaluation_protocol.json`).
- **Anomaly Score:** Standardized Euclidean distance across 16 bands:
  \[
  d_{\text{high}}(\mathbf{x}) = \sqrt{\sum_{i=1}^{16} \left(\frac{x_i - \text{median}_i}{\text{scale}_i}\right)^2}
  \]
- **Frozen Threshold:** \(\tau_{\text{high}} = 55.309466\) (calibrated to 5% FAR on calibration normals).
- **Decision Rule:** Anomaly (\(1\)) if \(d_{\text{high}}(\mathbf{x}) > \tau_{\text{high}}\); else Normal (\(0\)).
- **Frequency Estimator:** Hann-windowed rFFT peak frequency (\(\text{fft\_peak\_frequency}\)).

### 2.3 Adaptive Selector (Two-Stage Cascade)
- **First Stage:** Run Fixed-Low feature extraction and zero-crossing frequency.
- **Frozen Decision Boundaries (from calibration normals):**
  - \(\tau_{90} = 5.004375\) (90th percentile of normal calibration low scores)
  - \(\tau_{99} = 5.573239\) (99th percentile of normal calibration low scores)
  - \(f_{\text{zcf}, 1\%} = 183.96\) Hz, \(f_{\text{zcf}, 99\%} = 644.63\) Hz (1st and 99th percentiles of calibration normal ZCFs)
- **Decision & Escalation Logic:**
  1. If \(f_{\text{zcf}}(\mathbf{x}) < f_{\text{zcf}, 1\%}\) or \(f_{\text{zcf}}(\mathbf{x}) > f_{\text{zcf}, 99\%}\): Escalate to Stage 2 (`stage2_high`, `fft_used=True`, reason: `zcf_out_of_range`).
  2. Else if \(d_{\text{low}}(\mathbf{x}) \le \tau_{90}\): Stop at Stage 1 as Confidently Normal (`stage1_low`, `fft_used=False`, label: 0, reason: `confident_normal`).
  3. Else if \(d_{\text{low}}(\mathbf{x}) \ge \tau_{99}\): Stop at Stage 1 as Confidently Anomalous (`stage1_low`, `fft_used=False`, label: 1, reason: `confident_anomaly`).
  4. Else (\(\tau_{90} < d_{\text{low}}(\mathbf{x}) < \tau_{99}\)): Escalate to Stage 2 (`stage2_high`, `fft_used=True`, reason: `uncertain_score`).
- When escalated, Fixed-High predictions and FFT peak frequency estimates are adopted.

---

## 3. Evaluation Metrics and Standards

### 3.1 Binary Anomaly Detection
Evaluated on the binary normal (\(0\)) vs anomaly (\(1\)) task:
- **Precision:** \(\text{TP} / (\text{TP} + \text{FP})\)
- **Recall:** \(\text{TP} / (\text{TP} + \text{FN})\)
- **F1 Score:** \(2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}\)
- **False-Alarm Rate (FAR):** \(\text{FP} / (\text{FP} + \text{TN})\), calculated strictly over normal ground-truth windows.

### 3.2 Frequency-Estimation Error
- **Eligibility:** Scored exclusively on cases with a defined ground-truth fundamental frequency (`normal` and `frequency_shift`).
- **Exclusion:** Multi-tone (`added_harmonic`) and transient (`impact`) windows have no single target frequency and are excluded from frequency MAE.
- **Metric:** Mean Absolute Error (MAE) in Hz: \(\frac{1}{K}\sum |f_{\text{est}} - f_{\text{target}}|\).

### 3.3 Computational Work Accounting
Operation counts represent algorithmic operations and selector overhead (not physical energy):
- **Fixed-Low standalone:** \(10 N + 25 = 10,265\) ops per window (\(N=1024\)).
- **Fixed-High standalone:** \(2.5 N \log_2(N) + 3.5 N + 66 = 29,250\) ops per window (\(N=1024\)).
- **Adaptive Stage 1 alone:** \(10,265\) ops.
- **Adaptive Escalated (Stage 1 + Stage 2):** \(10,265 + 29,250 = 39,515\) ops.
- **Adaptive Weighted Average:**
  \[
  \text{Ops}_{\text{adaptive}} = \frac{N_{\text{stage1}} \times 10,265 + N_{\text{stage2}} \times 39,515}{N_{\text{total}}}
  \]
- **Python Execution Runtime:** Recorded per window using `time.perf_counter()` and reported separately in milliseconds.

### 3.4 Statistical Uncertainty and Group-Level Clustering
Because signal windows sharing the same source `group_id` are correlated across SNRs and cases:
- Group-level bootstrap resampling (1,000 iterations) will be performed by resampling source groups with replacement to construct 95% confidence intervals.
- The 20 held-out test groups represent a finite cluster sample; limitations of sample clustering will be documented alongside test metrics.
