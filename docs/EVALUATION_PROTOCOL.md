# Evaluation Protocol (v1.0.0 Historical Baseline & v1.1.0 Frozen Protocol)

> **Status:** PROTOCOL v1.1.0 FROZEN FOR PHASE 5 HELD-OUT EVALUATION  
> **Initial Freeze Version:** 1.0.0 (commit `0dc2b39`, `results/evaluation_protocol.json`)  
> **Phase 4 Investigation:** Identified Candidate Selector v1.1 (`cand_v1.1_p60_anomaly_first`) in `docs/PHASE4_INVESTIGATION.md`  
> **Active Evaluation Protocol Version:** 1.1.0 (frozen in `results/evaluation_protocol_v1.1.json`)  
> **Scope:** Preserves baseline methods and historical v1.0. Evaluates Candidate Selector v1.1 on held-out test data once without tuning.

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
- **Held-Out Test Split:** 20 source groups (\(320\) windows). Verifiable SHA-256 prefix: `b99e0b36ea5b6659`.
  - **Group IDs (20):** `['g0000', 'g0004', 'g0005', 'g0009', 'g0015', 'g0020', 'g0022', 'g0024', 'g0029', 'g0032', 'g0033', 'g0034', 'g0042', 'g0046', 'g0062', 'g0067', 'g0068', 'g0070', 'g0077', 'g0078']`.
  > **STRICT RULE:** The held-out test set is evaluated exactly once in Phase 5. No parameters, thresholds, features, or rules shall be tuned or altered based on test results.
- **Development Split:** 60 source groups (\(960\) windows). Verifiable SHA-256 prefix: `9ece130e3b4054a7`.
  - **Model-Fit Partition (30 groups, 480 windows; 120 normal windows):** `['g0001', 'g0002', 'g0010', 'g0011', 'g0013', 'g0017', 'g0021', 'g0023', 'g0025', 'g0026', 'g0028', 'g0030', 'g0038', 'g0039', 'g0040', 'g0041', 'g0045', 'g0047', 'g0048', 'g0050', 'g0052', 'g0054', 'g0056', 'g0058', 'g0061', 'g0064', 'g0074', 'g0075', 'g0076', 'g0079']`. Used solely for fitting reference feature medians and MAD scales.
  - **Threshold-Calibration Partition (30 groups, 480 windows; 120 normal windows):** `['g0003', 'g0006', 'g0007', 'g0008', 'g0012', 'g0014', 'g0016', 'g0018', 'g0019', 'g0027', 'g0031', 'g0035', 'g0036', 'g0037', 'g0043', 'g0044', 'g0049', 'g0051', 'g0053', 'g0055', 'g0057', 'g0059', 'g0060', 'g0063', 'g0065', 'g0066', 'g0069', 'g0071', 'g0072', 'g0073']`. Used solely for calibrating baseline thresholds and selector boundaries.

---

## 2. Frozen Methods and Decision Rules

### 2.1 Fixed-Low Baseline (Common to v1.0 and v1.1)
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
- **Cost:** \(10 N + 25 = 10,265\) ops per window (\(N=1024\)).

### 2.2 Fixed-High Baseline (Common to v1.0 and v1.1)
- **Features:** 16-band normalized spectral energies from Hann-windowed rFFT across 0–1200 Hz (75 Hz bandwidth per band, normalized to sum to 1.0).
- **Reference Parameters:** Fitted on model-fit normals (16-dimensional median and robust scale vectors; recorded in `results/evaluation_protocol_v1.1.json`).
- **Anomaly Score:** Standardized Euclidean distance across 16 bands:
  \[
  d_{\text{high}}(\mathbf{x}) = \sqrt{\sum_{i=1}^{16} \left(\frac{x_i - \text{median}_i}{\text{scale}_i}\right)^2}
  \]
- **Frozen Threshold:** \(\tau_{\text{high}} = 55.309466\) (calibrated to 5% FAR on calibration normals).
- **Decision Rule:** Anomaly (\(1\)) if \(d_{\text{high}}(\mathbf{x}) > \tau_{\text{high}}\); else Normal (\(0\)).
- **Frequency Estimator:** Hann-windowed rFFT peak frequency (\(\text{fft\_peak\_frequency}\)).
- **Cost:** \(2.5 N \log_2(N) + 3.5 N + 66 = 29,250\) ops per window (\(N=1024\)).

### 2.3 Historical v1.0 Adaptive Selector (Reference Protocol Baseline)
- **Architecture:** Two-stage cascade running Fixed-Low first.
- **Frozen Boundaries (from normal calibration data):**
  - \(\tau_{90} = 5.004375\), \(\tau_{99} = 5.573239\)
  - \(f_{\text{zcf}, 1\%} = 183.96\) Hz, \(f_{\text{zcf}, 99\%} = 644.63\) Hz
- **Decision Order:**
  1. If \(f_{\text{zcf}} < 183.96\) or \(f_{\text{zcf}} > 644.63\) \(\to\) Escalate to Fixed-High (`zcf_out_of_range`).
  2. Else if \(d_{\text{low}} \le 5.0044\) \(\to\) Stop at Stage 1 as Normal (`confident_normal`).
  3. Else if \(d_{\text{low}} \ge 5.5732\) \(\to\) Stop at Stage 1 as Anomaly (`confident_anomaly`).
  4. Else \(\to\) Escalate to Fixed-High (`uncertain_score`).
- **Historical Development Finding:** Escalated on 100% of impacts (due to ZCF check before anomaly check) while missing frequency shifts and added harmonics due to noise-inflated ZCF bounds.

### 2.4 Frozen Candidate Selector v1.1 (Protocol v1.1.0 Active Method)
- **Architecture:** Two-stage cascade implementing the corrected decision hierarchy ([`CandidateAdaptiveSelector`](file:///c:/Users/PC/Desktop/adaptive-acoustic-monitoring/python/candidate_selector.py)).
- **Frozen Boundaries (calibrated exclusively on the 30 threshold-calibration groups):**
  - \(p_{\text{anomaly}} = 7.140502\) (\(1.2 \times \max(d_{\text{low}})\) on normal calibration signals).
  - \(p_{\text{normal}} = 1.771420\) (60th percentile of normal calibration low scores).
  - \(z_{\min} = 173.59375\) Hz, \(z_{\max} = 263.90625\) Hz (calibrated on clean normal calibration signals \(\ge 20\) dB SNR with \(\pm 10\) Hz margin).
- **Exact Decision Hierarchy:**
  1. **Step 1 — Confident Anomaly Check:**
     If \(d_{\text{low}}(\mathbf{x}) \ge 7.140502\): Stop at Stage 1 as Anomaly (`stage1_low`, `label=1`, `fft_used=False`, `reason="confident_anomaly"`, cost: \(10,265\) ops).
     *Effect:* 100% of impact anomalies stop cheaply at Stage 1 without paying FFT escalation overhead.
  2. **Step 2 — Frequency Discrepancy Check:**
     Else if \(f_{\text{zcf}}(\mathbf{x}) < 173.59375\) or \(f_{\text{zcf}}(\mathbf{x}) > 263.90625\): Escalate to Fixed-High (`stage2_high`, `fft_used=True`, `reason="zcf_out_of_range"`, cost: \(39,515\) ops).
     *Effect:* Routes clean frequency shifts and noise-smeared low-SNR windows to full spectral analysis.
  3. **Step 3 — Confident Normal Check:**
     Else if \(d_{\text{low}}(\mathbf{x}) \le 1.771420\): Stop at Stage 1 as Normal (`stage1_low`, `label=0`, `fft_used=False`, `reason="confident_normal"`, cost: \(10,265\) ops).
  4. **Step 4 — Uncertain Score Check:**
     Else (\(1.771420 < d_{\text{low}}(\mathbf{x}) < 7.140502\)): Escalate to Fixed-High (`stage2_high`, `fft_used=True`, `reason="uncertain_score"`, cost: \(39,515\) ops).
- **Inference-Time Input Guarantee:** Accepts strictly \(x \in \mathbb{R}^{1024}\). Does NOT access `snr_db`, `group_id`, `case`, or nominal \(f_0\).
- **Candidate Feature Clarification:** Does NOT evaluate \(R_{\text{diff}}\) at inference time; all operations are contained within the \(10,265\) Stage 1 figure.

---

## 3. Evaluation Metrics and Standards

### 3.1 Binary Anomaly Detection
- **Precision:** \(\text{TP} / (\text{TP} + \text{FP})\)
- **Recall:** \(\text{TP} / (\text{TP} + \text{FN})\)
- **F1 Score:** \(2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}\)
- **False-Alarm Rate (FAR):** \(\text{FP} / (\text{FP} + \text{TN})\), calculated strictly over normal ground-truth windows.

### 3.2 Frequency-Estimation Error
- **Eligibility:** Scored exclusively on cases with a defined ground-truth fundamental frequency (`normal` and `frequency_shift`).
- **Exclusion:** Multi-tone (`added_harmonic`) and transient (`impact`) windows have no single target frequency and are excluded from frequency MAE.
- **Metric:** Mean Absolute Error (MAE) in Hz relative to `target_frequency_hz`.

### 3.3 Computational Work Accounting
- **Fixed-Low standalone:** \(10,265\) ops per window.
- **Fixed-High standalone:** \(29,250\) ops per window.
- **Adaptive Stage 1 alone:** \(10,265\) ops per window.
- **Adaptive Escalated (Stage 1 + Stage 2):** \(10,265 + 29,250 = 39,515\) ops per window.
- **Weighted Average Ops:** \(\frac{N_{\text{stage1}} \times 10,265 + N_{\text{stage2}} \times 39,515}{N_{\text{total}}}\).
- **Savings vs. Fixed-High:** \(\frac{29,250 - \text{Weighted Ops}}{29,250} \times 100\%\).
- **Break-Even Limit:** Positive savings requires escalation rate \(E < \frac{29,250 - 10,265}{29,250} = \mathbf{64.91\%}\).
- **Desktop Runtime:** Measured per window using `time.perf_counter()` and reported separately in milliseconds. Neither operation counts nor runtime represent physical ESP32 energy.

### 3.4 Group-Level Bootstrap Confidence Intervals
To account for clustering across the 20 source groups in the test set:
- **Resampling Unit:** Entire source groups (all 16 windows per group resampled together).
- **Iterations:** 1,000 bootstrap iterations.
- **Random Seed:** `20261013`.
- **Interval:** Empirical 2.5th and 97.5th percentiles (95% bootstrap CI).

---

## 4. Pre-Test Verification Summary

Before opening test data, development checks confirmed:
- Fixed baselines fit/calibrate cleanly on development partitions.
- 5-fold grouped cross-validation independently selected `cand_v1.1_p60_anomaly_first` over all 12 candidate configurations.
- All 28 project unit tests pass.
- Test split hash verified: `b99e0b36ea5b6659`.
