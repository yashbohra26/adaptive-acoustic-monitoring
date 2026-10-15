# Adaptive acoustic monitoring with a two-stage signal-processing cascade

*First manuscript draft for the progress demonstration and Review 3 development. This draft reports repository results and is not a publication-ready paper.*

**Authors and affiliations:** to be supplied by the project team.

## Abstract

We implemented a two-stage detector that computes time-domain features first and escalates selected windows to spectral analysis. We compared it with fixed-low and fixed-high detectors on grouped, synthetic machine-like signals spanning four cases and four signal-to-noise ratios. On the Phase 5 holdout, Candidate v1.1 reached F1 0.9170, recall 0.8750, and false-alarm rate 0.1000, with an estimated cost of 26,992.3 operations per window (7.72% below the Fixed-High proxy). Its recall did not meet the same-evaluation comparison margin. In a separate evaluation, Candidate v1.2 matched Fixed-High's point estimates for F1 (0.9534) and recall (0.9375), with 1.78% lower estimated work. Its 95% group-bootstrap interval for operation savings ranged from −1.66% to 4.91%, including zero. The project has not resolved whether final scientific acceptance uses fixed Phase 4 cutoffs or margins calculated against Fixed-High within each evaluation. These results describe software tested on controlled synthetic signals. They do not establish real-machine performance or energy savings; neither physical-device energy nor live acoustic input was measured.

## Keywords

Acoustic monitoring; anomaly detection; adaptive computation; cascaded signal processing; synthetic signals; embedded systems; operation-count proxy

## 1. Introduction

Machine faults can change a sound's level, periodicity, or frequency content. This project tests whether a detector can use inexpensive time-domain features to reject clear cases and reserve spectral analysis for windows that need it. The comparison includes a fixed-low detector, which always uses time-domain features, and a fixed-high detector, which always computes spectral features.

We implemented the cascade in Python and ported its decision logic to a desktop C++ harness. We evaluated detection, frequency estimation where a target frequency exists, and an operation-count proxy on controlled synthetic windows. The experiment asks whether conditional spectral processing can reduce estimated work while retaining detection quality on held-out synthetic conditions. It does not test generalization to operating machinery: the project has no machine recordings, live microphone stream, or physical ESP32 measurements.

## 2. Research problem and objectives

The study has four practical objectives:

1. Implement fixed-low and fixed-high anomaly detectors with development-only fitting and threshold calibration.
2. Add an adaptive policy that performs low-cost analysis first and escalates uncertain or frequency-discrepant windows.
3. Compare detection quality, frequency-estimation error where a target is defined, and an algorithmic work proxy on grouped held-out synthetic signals.
4. Port representative decision logic to C++ and verify the software output against Python test vectors.

The study is exploratory. The project record leaves the final scientific acceptance rule undecided, so results are reported against both documented interpretations. Neither is designated here as authoritative.

## 3. Related work

Classical acoustic diagnosis often begins with statistical features or machine-specific signal analysis. Heng and Nor studied crest factor, kurtosis, and related statistics for bearing defects, and found that shaft speed affected their measurements [3]. Germen et al. used multi-microphone correlation and wavelet features with self-organizing maps for induction-motor fault classification [6]. Later work compares physically motivated features with MFCCs in an industrial machine hall [11], uses timbral and short-term features with an SVM on MIMII [12], and develops acoustic diagnosis for an air compressor with eight operating states [18]. These studies support acoustic features as practical diagnostic inputs, while showing that machine speed, background noise, microphone placement, and task definition affect results. Their machine-specific recordings and fault-classification tasks differ from our grouped synthetic anomaly-detection experiment.

Unsupervised and semi-supervised ASD papers use learned representations, reconstruction errors, classification confidence, or statistical density models. Examples include ArcFace with a GMM [1], an autoencoder objective based on a Neyman–Pearson false-positive constraint [21], classifier-based methods [13], and identity-conditioned autoencoders [7]. Other work uses Transformers [8], convolutional autoencoders [5, 9], a fully connected U-Net [19], GAN reconstruction [15], spatial deep features [16], or selected spectrogram frames and frequency dimensions [2, 10]. These papers report results on DCASE or MIMII benchmarks, but their training assumptions, input features, split designs, and reported metrics vary. They do not provide a single directly comparable baseline for this project. ToyADMOS and MIMII are established machine-sound datasets [25, 26]; neither is used in the present evaluation.

Several papers address changes in operating state or environment. Xu et al. first identify hydro-turbine load changes, then apply PCA and adaptive K-means under stable load [4]. Zhang et al. test diesel-generator recordings across loads [5], while Chen et al. and Zhang et al. study domain-shift handling in learned detectors [9, 10]. Reviews by Jombo and Zhang and by Qurthobi et al. describe noise and operating-condition variability as persistent challenges [17, 20]. These findings motivate preserving source groups and varying SNR in our synthetic protocol. The generator does not represent the machine, load, room, and sensor variation found in these real-recording studies.

Computational cost is measured inconsistently across this literature. Sigtia et al. compare recognition performance with model cost for environmental sound recognition, rather than machine anomaly detection [14]. Wang et al. and Zhang et al. report model sizes of 0.08 MB and 0.12 MB for their ASD methods [2, 10], and Di Fiore et al. discuss inference time and memory [7]. Such measures do not equal per-window arithmetic counts or physical energy. Xu et al. describe one-second online hydro-turbine updates [4], but this timing claim is also not a hardware-energy comparison with our detector. Prior work on cascades and early exits provides a broader conditional-computation analogy [22, 23]; the machinery-diagnostics background also informs the low-stage statistical features [24].

Across the papers supplied for this review, we did not find a directly matched comparison combining an interpretable time-domain first stage, conditional spectral escalation, fixed-low and fixed-high references, and per-window work accounting on grouped held-out synthetic signals. This observation is limited to the supplied set and does not establish that such work is absent elsewhere. Our contribution is an implementation and a controlled comparison of this design, including its recall failures and uncertain cost savings. The literature review remains preliminary rather than systematic.
## 4. System architecture and methodology

Every detector processes a 1,024-sample window sampled at 4 kHz (0.256 s). Fixed-Low extracts log RMS, log crest factor, and log kurtosis. It compares those features with robust statistics fitted on normal development windows using a standardized Euclidean distance, and estimates frequency from zero crossings. Fixed-High computes normalized energy in 16 bands spanning 0 to 1,200 Hz from a Hann-windowed real FFT. It applies the same distance-based comparison to fitted spectral features and uses the FFT peak for its frequency estimate.

Reference medians and robust scales come from normal windows in the model-fit partition. We calibrated baseline decision thresholds on a separate set of normal threshold-calibration windows, targeting a 5% false-alarm rate. Candidate selector boundaries also use development calibration data. At inference, the policy receives the signal window; it does not use held-out labels, machine or source-group identifiers, the nominal fundamental, or SNR.

Fixed-High reuses one rFFT result to compute both band energies and peak frequency. Tests check the FFT count and operation-accounting invariants. A desktop C++ harness compares decisions, selected stages, escalation reasons, operation proxies, and scores against five development vectors. This establishes parity for those vectors only. It does not validate the proxy as a model of hardware energy use.

## 5. Synthetic signal generation and experimental design

Phase 5 contains 80 source groups and 1,280 windows. Each source recipe contributes 16 windows: four signal cases at four SNR levels. All 16 stay in the same split. The cases are:

- **Normal:** a sinusoidal tone with a nominal fundamental sampled from 180 to 260 Hz.
- **Frequency shift:** a sinusoid whose fundamental is multiplied by a factor from 1.35 to 1.60.
- **Added harmonic:** a fundamental plus a second harmonic at 0.45 relative amplitude; there is no single target frequency for this multi-tone case.
- **Impact:** a decaying sinusoidal transient with a 0.009 s decay constant and a carrier sampled from 500 to 900 Hz; it has no stable fundamental target.

The generator adds white Gaussian noise at 30, 20, 10, or 0 dB SNR. Normal windows are labeled non-anomalous; the other three cases are labeled anomalous. Configured random seeds and group-level metadata support reproducibility.

Under Protocol 1.1.0, 60 groups (960 windows) form the development set and 20 groups (320 windows) the held-out test set. The development groups are split into 30 model-fit and 30 threshold-calibration groups. Phase 5 uses master seed 20261013; the held-out group-name hash begins `b99e0b36ea5b6659`. The frozen protocol permits one evaluation of this holdout and bars tuning against its results.

Candidate v1.2 uses a separate evaluation of 20 groups and 320 windows, generated with seed 20261014. Its provenance record gives canonical content hash `962643581c21c335677cc41418e21c0bbb07acc14cdf0ad1356eee373d2446ac`. We report the two evaluations separately and do not pool their windows or results. The 128-window preview in `work/local_build_preview` is for smoke testing only and is not evaluation evidence.

## 6. Baselines and adaptive policies

### 6.1 Fixed baselines

Fixed-Low computes time-domain features and a zero-crossing estimate for every window; its cost proxy is 10,265 operations per window. Fixed-High computes the 16-band representation and frequency estimate for every window; its proxy is 29,250 operations per window. Both thresholds are calibrated on development data.

### 6.2 Historical adaptive v1.0

Adaptive v1.0 begins with Fixed-Low. It escalates when zero-crossing frequency falls outside the 1st to 99th percentile range from normal calibration, or when the low-stage score lies between the 90th and 99th percentiles. Escalated windows use the high-stage decision and frequency estimate. In development, checking frequency first sent every impact case to the high stage, including cases whose low-stage score already indicated an anomaly.

### 6.3 Candidate v1.1

Candidate v1.1 tests the low-stage score before the frequency bounds. A sufficiently high score produces an anomaly decision immediately. Otherwise, an out-of-range zero-crossing frequency triggers escalation; a sufficiently low score produces a normal decision; all remaining windows go to Fixed-High. The frozen boundaries are $p_{anomaly}=7.140502$, $p_{normal}=1.771420$, $z_{min}=173.59375$ Hz, and $z_{max}=263.90625$ Hz. Here, $p_{normal}$ is the 60th percentile of normal calibration scores. The clean-frequency bounds use calibration signals at or above 20 dB SNR with a 10 Hz margin.

### 6.4 Candidate v1.2

For Protocol 1.2.0, development-only grouped cross-validation followed the v1.1 harmonic-miss analysis. It changed the normal-score percentile from 60 to 50, freezing $p_{normal}=1.311028$; the other candidate boundaries stayed the same. The resulting evaluation is reported separately from Phase 5.

## 7. Evaluation metrics and computational-cost model

We calculate precision as $TP/(TP+FP)$, recall as $TP/(TP+FN)$, F1 as their harmonic mean, and FAR as $FP/(FP+TN)$ over normal windows. Frequency MAE uses only normal and frequency-shift windows, for which the generator defines a target. We exclude added-harmonic and impact windows because neither has a single stable target frequency.

The work model assigns Stage 1 a proxy cost of

\[
C_1 = 10N + 25 = 10{,}265 \quad (N=1{,}024),
\]

and the Fixed-High spectral stage an additional

\[
C_2 = 2.5N\log_2(N) + 3.5N + 66 = 29{,}250.
\]

An escalated window incurs 39,515 proxy operations across both stages. Mean adaptive cost depends on the observed proportion of windows that escalate. We calculate relative savings as $(29{,}250-C_{avg})/29{,}250\times100\%$. Under this model, savings require an escalation rate below 64.91%. Operation counts are estimates under the project's accounting rules; they are not measurements of instruction count, power, energy, battery life, or ESP32 runtime.

For Phase 5, the saved 95% intervals use 1,000 group-bootstrap resamples and seed 20261013. The v1.2 provenance record uses 1,000 resamples and seed 20261014. Each resample draws complete source groups, keeping the 16 related variants together. The intervals are method-specific; their overlap does not establish equivalence.

We report desktop runtime separately from the operation proxy. The common benchmark uses five warmups and 25 timed repetitions per window with `time.perf_counter_ns()`. These measurements reflect Python/NumPy on the host; they do not measure embedded runtime or energy.

## 8. Results

### 8.1 Phase 5 held-out evaluation (Protocol 1.1.0)

Table 1 gives the saved Phase 5 estimates. The result JSON reports frequency MAE for all four methods, but the historical per-window CSV omits adaptive v1.0 and v1.1 frequency estimates. Those two MAEs therefore cannot be independently recalculated from the saved window records.

| Method | F1 (95% group CI) | Recall (95% group CI) | FAR | Frequency MAE (Hz) | Mean ops proxy | Savings vs Fixed-High |
|---|---:|---:|---:|---:|---:|---:|
| Fixed-Low | 0.5572 (0.5313–0.5788) | 0.3958 (0.3750–0.4167) | 0.0750 | 91.44 | 10,265.0 | 64.91% |
| Fixed-High | 0.9492 (0.9123–0.9772) | 0.9333 (0.8667–0.9833) | 0.1000 | 0.85 | 29,250.0 | 0.00% |
| Adaptive v1.0 | 0.6243 (0.6023–0.6431) | 0.4708 (0.4458–0.4917) | 0.1125 | 38.37 | 21,416.6 | 26.78% |
| Candidate v1.1 | 0.9170 (0.8662–0.9575) | 0.8750 (0.7957–0.9458) | 0.1000 | 2.75 | 26,992.3 | 7.72% |

*Table 1. Phase 5 metrics from the 320-window Protocol 1.1.0 holdout. Confidence intervals are the saved 95% group-bootstrap intervals. Work and savings are operation-count proxies. Source: [frozen evaluation JSON](../results/phase5_final_evaluation.json); per-window records: [historical prediction CSV](../results/phase5_window_predictions.csv).*

Candidate v1.1 detected all impact windows at Stage 1, with no escalation. It detected 66 of 80 added-harmonic windows (82.5%), compared with 100% of impacts and 80% of frequency shifts. The saved analysis locates the 14 missed harmonic windows mainly among subtle, high-SNR examples. Overall recall was 0.8750, below the same-evaluation floor of 0.8833. Software tests can verify that this failure is reported; they cannot turn it into a scientific pass.

The separate benchmark measured mean desktop runtime per window at 0.162 ms for Fixed-Low, 0.468 ms for Fixed-High, 0.447 ms for Candidate v1.1, and 0.374 ms for historical v1.0. All four used the same timing harness. These are host-runtime observations, not hardware results.

### 8.2 Independent evaluation (Protocol 1.2.0)

| Method | F1 (95% group CI) | Recall (95% group CI) | FAR | Mean ops proxy | Point savings vs Fixed-High |
|---|---:|---:|---:|---:|---:|
| Fixed-High | 0.9534 (0.9231–0.9793) | 0.9375 (0.8792–0.9833) | 0.0875 | 29,250.0 | 0.00% |
| Candidate v1.1 | 0.9047 (0.8767–0.9325) | 0.8500 (0.8042–0.8958) | 0.0875 | 26,169.7 | 10.53% |
| Candidate v1.2 | 0.9534 (0.9231–0.9793) | 0.9375 (0.8792–0.9833) | 0.0875 | 28,729.1 | 1.78% |

*Table 2. Saved point estimates and 95% group-bootstrap intervals from the separate 320-window evaluation generated with seed 20261014. The independent result file does not report frequency MAE. Source: [frozen evaluation JSON](../results/candidate_v1.2_independent_evaluation.json) and [provenance and uncertainty record](../results/candidate_v1.2_provenance_and_uncertainty.json).*

On this separate holdout, Candidate v1.1 again missed the same-evaluation recall floor of 0.8875. Candidate v1.2's point estimates for F1 and recall matched Fixed-High, as did its FAR, while its operation proxy was lower. The 95% group-bootstrap interval for v1.2 savings runs from −1.66% to 4.91% and includes zero. It does not establish positive savings or equivalence. The saved artifact contains no paired interval for quality differences.

### 8.3 Acceptance-rule implications

Phase 4 records fixed cutoffs of F1 ≥ 0.90435, recall ≥ 0.87917, FAR ≤ 0.06417, and operation proxy < 29,250. It also describes screening margins relative to Fixed-High: F1 and recall may be at most 0.05 lower, FAR at most 0.01 higher, and operation proxy must remain < 29,250. The frozen v1.1 and v1.2 protocols do not say which rule governs the final scientific claim. Tests and the v1.2 result status calculate the same-evaluation margins, but that implementation detail does not settle the research decision.

| Evaluation | Candidate | Fixed Phase 4 cutoffs (F1 / recall / FAR / ops) | Same-evaluation Fixed-High margins (F1 / recall / FAR / ops) |
|---|---|---|---|
| Phase 5 | v1.1 | Pass / **Fail** / **Fail** / Pass | Pass / **Fail** / Pass / Pass |
| Independent | v1.1 | Pass / **Fail** / **Fail** / Pass | Pass / **Fail** / Pass / Pass |
| Independent | v1.2 | Pass / Pass / **Fail** / Pass | Pass / Pass / Pass / Pass |

*Table 3. Threshold outcomes computed from saved values under each interpretation. The table documents the disagreement; it does not designate either rule as the final gate.*

## 9. Discussion and failure analysis

The Phase 5 case results point to added harmonics as a main source of Candidate v1.1's recall loss. The policy can stop at Stage 1 when its score falls within the confident-normal region, before spectral analysis exposes a weaker harmonic change. It saved 7.72% of the Fixed-High operation proxy, and its F1 was within 0.05 of Fixed-High, but its recall missed the comparison margin by 0.0083.

The earlier v1.0 policy shows that lower estimated work did not guarantee useful detection: it saved 26.78% of the proxy but reached recall 0.4708. Candidate v1.1 improved recall while saving less. Candidate v1.2 matched Fixed-High's point estimates on its own holdout after changing the normal-score boundary. Its savings remain uncertain, and the separately generated v1.1 and v1.2 holdouts do not support a paired comparison.

The cost model assigns fixed arithmetic estimates to feature and FFT processing. It compares branches under the project's accounting rules, but omits memory access, compiler optimization, peripheral activity, and measured energy. Desktop timings describe this Python implementation and cannot estimate ESP32 latency.

## 10. Threats to validity and limitations

### Signal realism

All windows use controlled waveforms with additive white Gaussian noise. The generator does not model room acoustics, reverberation, changing machine loads, sensor placement, multiple sources, clipping, or the diversity of faults found in industrial recordings.

### Sampling and generalization

Each holdout contains 20 synthetic source groups. Its group-bootstrap intervals describe variation across those generated groups; they do not represent variation across real machines or factories. The v1.2 holdout is separate from Phase 5 but uses the same controlled signal-generation design.

### Acceptance protocol

The project record includes fixed Phase 4 cutoffs and same-evaluation Fixed-High margins, but designates neither as the final rule. The research team and supervisor need to decide which one governs any final pass/fail claim.

### Metric reproducibility

The Phase 5 JSON records frequency MAE, but its frozen window CSV omits the adaptive v1.0 and v1.1 estimates needed to recalculate those values. Future Phase 5 outputs use a separate schema and filenames with all four estimates and explicit eligible and missing counts. These fields do not establish the historical values independently. The v1.2 evaluation artifact has no frequency-MAE result.

### Software and hardware validation

Python tests and the desktop C++ harness validate software behavior. The harness uses five development vectors and checks decisions, stages, escalation reasons, work proxies, and score tolerance. Interactive Wokwi execution still needs manual confirmation. The project has not tested a physical ESP32 or microphone, streamed live audio, or measured electrical power or energy savings.

### Provenance

The protocol files record signal settings, group splits, and seeds. The v1.2 provenance record also includes a canonical content hash and bootstrap intervals. The historical Phase 5 result has no Git revision or Python-environment record. Provenance fields added for future runs cannot recover that missing history.

## 11. Conclusion and future work

The implementation compares fixed-low, fixed-high, and adaptive processing on controlled synthetic signals. Candidate v1.1 saved estimated work in Phase 5 but missed the recall margin. Candidate v1.2 matched Fixed-High's point metrics on a separate holdout, although its savings interval includes zero and its FAR exceeds the fixed Phase 4 cutoff. Since the final acceptance rule remains undecided, the project cannot yet make a binary scientific acceptance claim.

For Review 3, the team needs to settle that rule and strengthen the literature comparison. If project scope permits, evaluation on real or public machine recordings would test whether the observed behavior carries beyond these generated signals. Physical ESP32 timing and energy measurements, live microphone acquisition, and manual end-to-end Wokwi verification also remain to be done. Any new confirmatory study needs a defined protocol and data boundary before results are examined.

## 12. Work to complete before Review 3

- Obtain supervisor/researcher approval for the final scientific acceptance rule and state it prospectively for any new confirmatory study.
- Expand and verify the literature review, especially comparisons with adaptive classical DSP and resource-aware embedded monitoring. This manuscript does not claim a systematic search.
- Decide whether the next study can use real machine recordings or established public datasets, and define the data split before evaluating candidates.
- Reproduce adaptive frequency MAE from retained per-window estimates in future runs; the historical Phase 5 CSV lacks those two adaptive estimate columns.
- Complete interactive Wokwi verification and, if claims about deployment efficiency are desired, measure execution time and energy on physical target hardware.
- Record author names/affiliations, intended venue, and any institutional or project reporting requirements.

### Saved figures available for integration

The manuscript's figures can be drawn from the repository's saved outputs. The Phase 5 plots are generated by `python/plot_evaluation_results.py`; the independent v1.2 plots are generated by `python/plot_v1_2_results.py`; and the acceptance-rule comparison plot is generated by `python/plot_protocol_audit.py`.

- Figure 1. Phase 5 overall performance: [PNG](../results/figures/fig1_overall_performance.png).
- Figure 2. Phase 5 computational work and desktop runtime: [PNG](../results/figures/fig2_work_and_runtime.png).
- Figure 3. Phase 5 recall by case and SNR: [PNG](../results/figures/fig3_case_and_snr_recall.png).
- Figure 4. Candidate v1.2 independent performance with group-bootstrap intervals: [PNG](../results/figures/fig1_v1_2_independent_performance.png).
- Figure 5. Independent v1.2 cost and escalation: [PNG](../results/figures/fig2_v1_2_work_and_escalation.png).
- Figure 6. Acceptance-rule interpretations plotted against saved recall and work values: [PNG](../results/figures/protocol_audit/audit_cost_vs_recall.png).

## 13. References

1. J. Wu, F. Yang, and W. Hu, “Unsupervised anomalous sound detection for industrial monitoring based on ArcFace classifier and Gaussian mixture model,” *Applied Acoustics*, vol. 203, 109188, 2023. https://doi.org/10.1016/j.apacoust.2022.109188.
2. Y. Wang, Q. Zhang, W. Zhang, and Y. Zhang, “A lightweight framework for unsupervised anomalous sound detection based on selective learning of time-frequency domain features,” *Applied Acoustics*, vol. 228, 110308, 2025. https://doi.org/10.1016/j.apacoust.2024.110308.
3. R. B. W. Heng and M. J. M. Nor, “Statistical analysis of sound and vibration signals for monitoring rolling element bearing condition,” *Applied Acoustics*, vol. 53, nos. 1–3, pp. 211–226, 1998.
4. X. Xu, H. Wen, H. Lin, Z. Li, and C. Huang, “Online detection method for variable load conditions and anomalous sound of hydro turbines using correlation analysis and PCA-adaptive-K-means,” *Measurement*, vol. 224, 113846, 2024. https://doi.org/10.1016/j.measurement.2023.113846.
5. S. Zhang, J. Zhang, S. Shan, Y. Shen, C. Li, S. Zhu, K. Zhang, and H. Wei, “Diesel generator set anomalous sound detection based on lightweight convolutional autoencoder and prior knowledge integration,” *Measurement*, vol. 243, 116392, 2025. https://doi.org/10.1016/j.measurement.2024.116392.
6. E. Germen, M. Başaran, and M. Fidan, “Sound based induction motor fault diagnosis using Kohonen self-organizing map,” *Mechanical Systems and Signal Processing*, vol. 46, pp. 45–58, 2014.
7. E. Di Fiore, A. Ferraro, A. Galli, V. Moscato, and G. Sperlì, “An anomalous sound detection methodology for predictive maintenance,” *Expert Systems with Applications*, vol. 209, 118324, 2022. https://doi.org/10.1016/j.eswa.2022.118324.
8. J. Bai, J. Chen, M. Wang, M. S. Ayub, and Q. Yan, “SSDPT: Self-supervised dual-path transformer for anomalous sound detection,” *Digital Signal Processing*, vol. 135, 103939, 2023. https://doi.org/10.1016/j.dsp.2023.103939.
9. S. Chen, Y. Sun, J. Wang, M. Wan, M. Liu, and X. Li, “A multi-scale dual-decoder autoencoder model for domain-shift machine sound anomaly detection,” *Digital Signal Processing*, vol. 156, 104813, 2025. https://doi.org/10.1016/j.dsp.2024.104813.
10. Y. Zhang, J. Feng, Q. Zhang, J. Hu, and W. Zhang, “An unsupervised anomalous sound detection method based on similarity-driven automatic feature selection,” *Digital Signal Processing*, vol. 161, 105123, 2025. https://doi.org/10.1016/j.dsp.2025.105123.
11. C. Pichler, M. Neumayer, B. Schweighofer, C. Feilmayr, S. Schuster, and H. Wegleiter, “Acoustic-based detection technique for identifying worn-out components in large-scale industrial machinery,” *IEEE Sensors Letters*, vol. 7, no. 9, article 6006204, 2023. https://doi.org/10.1109/LSENS.2023.3307103.
12. Y. Ota and M. Unoki, “Anomalous sound detection for industrial machines using acoustical features related to timbral metrics,” *IEEE Access*, vol. 11, 2023. https://doi.org/10.1109/ACCESS.2023.3294334.
13. Y. Wang, Y. Zheng, Y. Zhang, Y. Xie, S. Xu, Y. Hu, and L. He, “Unsupervised anomalous sound detection for machine condition monitoring using classification-based methods,” *Applied Sciences*, vol. 11, no. 23, 11128, 2021. https://doi.org/10.3390/app112311128.
14. S. Sigtia, A. M. Stark, S. Krstulović, and M. D. Plumbley, “Automatic environmental sound recognition: Performance versus computational cost,” *IEEE/ACM Transactions on Audio, Speech, and Language Processing*, vol. 24, no. 11, pp. 2096–2107, 2016. https://doi.org/10.1109/TASLP.2016.2592698.
15. Y. Tagawa, R. Maskeliūnas, and R. Damaševičius, “Acoustic anomaly detection of mechanical failures in noisy real-life factory environments,” *Electronics*, vol. 10, no. 19, 2329, 2021. https://doi.org/10.3390/electronics10192329.
16. I. Thoidis, M. Giouvanakis, and G. Papanikolaou, “Semi-supervised machine condition monitoring by learning deep discriminative audio features,” *Electronics*, vol. 10, no. 20, 2471, 2021. https://doi.org/10.3390/electronics10202471.
17. G. Jombo and Y. Zhang, “Acoustic-based machine condition monitoring—Methods and challenges,” *Eng*, vol. 4, pp. 47–79, 2023. https://doi.org/10.3390/eng4010004.
18. N. K. Verma, R. K. Sevakula, S. Dixit, and A. Salour, “Intelligent condition based monitoring using acoustic signals for air compressors,” *IEEE Transactions on Reliability*, vol. 65, no. 1, pp. 291–309, 2016. https://doi.org/10.1109/TR.2015.2459684.
19. H. V. Truong, N. C. Hieu, P. N. Giao, and N. X. Phong, “Unsupervised detection of anomalous sound for machine condition monitoring using fully connected U-Net,” *Journal of ICT Research and Applications*, vol. 15, no. 1, pp. 41–55, 2021. https://doi.org/10.5614/itbj.ict.res.appl.2021.15.1.3.
20. A. Qurthobi, R. Maskeliūnas, and R. Damaševičius, “Detection of mechanical failures in industrial machines using overlapping acoustic anomalies: A systematic literature review,” *Sensors*, vol. 22, no. 10, 3888, 2022. https://doi.org/10.3390/s22103888.
21. Y. Koizumi, S. Saito, H. Uematsu, Y. Kawachi, and N. Harada, “Unsupervised detection of anomalous sound based on deep learning and the Neyman–Pearson lemma,” *IEEE/ACM Transactions on Audio, Speech, and Language Processing*, vol. 27, no. 1, pp. 212–223, 2019. https://doi.org/10.1109/TASLP.2018.2877258.
22. P. Viola and M. Jones, “Rapid object detection using a boosted cascade of simple features,” in *Proceedings of the 2001 IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR 2001)*, vol. 1, pp. I-511–I-518, 2001. https://doi.org/10.1109/CVPR.2001.990517.
23. S. Teerapittayanon, B. McDanel, and H. T. Kung, “BranchyNet: Fast inference via early exiting from deep neural networks,” in *23rd International Conference on Pattern Recognition (ICPR 2016)*, pp. 2464–2469, 2016. https://doi.org/10.1109/ICPR.2016.7900006.
24. R. B. Randall and J. Antoni, “Rolling element bearing diagnostics—A tutorial,” *Mechanical Systems and Signal Processing*, vol. 25, no. 2, pp. 485–520, 2011. https://doi.org/10.1016/j.ymssp.2010.07.017.
25. Y. Koizumi, S. Saito, H. Uematsu, N. Harada, and K. Imoto, “ToyADMOS: A Dataset of Miniature-Machine Operating Sounds for Anomalous Sound Detection,” in *2019 IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA 2019)*, pp. 313–317, 2019. https://doi.org/10.1109/WASPAA.2019.8937164.
26. H. Purohit et al., “MIMII Dataset: Sound Dataset for Malfunctioning Industrial Machine Investigation and Inspection,” in *Proceedings of the Detection and Classification of Acoustic Scenes and Events 2019 Workshop (DCASE 2019)*, pp. 209–213, 2019. https://arxiv.org/abs/1909.09347.