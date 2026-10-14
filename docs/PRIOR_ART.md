# Prior Art and Literature Review: Adaptive Signal Processing for Acoustic Condition Monitoring

## 1. Executive Summary

This document reviews primary research literature and authoritative technical references foundational to adaptive and multi-stage signal processing for acoustic condition monitoring. It outlines core principles from prior art, contrasts them with the methods evaluated in this project, and explicitly delineates external published findings from our experimental conclusions.

---

## 2. Core Literature and Authoritative References

### 2.1 The Attentional Rejection Cascade
* **Primary Reference:** Paul Viola and Michael Jones, "Rapid object detection using a boosted cascade of simple features," in *Proceedings of the 2001 IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR 2001)*, vol. 1, pp. I-511–I-518, 2001.  
  DOI: [`10.1109/CVPR.2001.990517`](https://doi.org/10.1109/CVPR.2001.990517)
* **Core Contribution:** Introduced the principle of an asymmetric cascade of classifiers arranged in stages of increasing computational complexity. Simpler early stages discard the vast majority of negative (background) instances with minimal work, preserving high recall while reserving expensive features only for ambiguous or potential positive candidates.
* **Relationship to this Project:** Our adaptive architecture adapts the attentional cascade concept for continuous 1D acoustic streaming. Stage 1 executes inexpensive time-domain feature extraction ($10,265$ ops proxy), rapidly classifying confident normal and obvious transient impact windows, while escalating to Stage 2 ($39,515$ cumulative ops proxy) only when zero-crossing frequency or low-score uncertainty demands full spectral analysis.

### 2.2 Dynamic Computation and Early-Exit Architectures
* **Primary Reference:** Surat Teerapittayanon, Bradley McDanel, and H. T. Kung, "BranchyNet: Fast inference via early exiting from deep neural networks," in *23rd International Conference on Pattern Recognition (ICPR 2016)*, pp. 2464–2469, 2016.  
  DOI: [`10.1109/ICPR.2016.7900006`](https://doi.org/10.1109/ICPR.2016.7900006)
* **Core Contribution:** Formalized dynamic computation with early exit branches from intermediate layers based on prediction confidence, reducing average inference latency in neural architectures.
* **Relationship and Contrast:** While BranchyNet and modern deep early-exit architectures target neural networks running on GPUs or edge accelerators (requiring hundreds of thousands to millions of multiply-accumulate operations per window), our project is motivated by resource-constrained microcontroller targets (such as an ESP32 or Cortex-M class device). We explore dynamic early exits using classical, interpretable statistical moments and single-pass rFFT spectral bins without deep neural weights. However, physical memory footprint and electrical energy dissipation remain unmeasured on hardware and serve as design motivations rather than demonstrated hardware properties.

### 2.3 Physics-Informed Machinery Diagnostics
* **Primary Reference:** R. B. Randall and J. Antoni, "Rolling element bearing diagnostics—A tutorial," *Mechanical Systems and Signal Processing*, vol. 25, no. 2, pp. 485–520, Feb. 2011.  
  DOI: [`10.1016/j.ymssp.2010.07.017`](https://doi.org/10.1016/j.ymssp.2010.07.017)
* **Core Contribution:** Established the fundamental domain physics of rotating machine diagnostics: statistical time-domain moments (RMS, crest factor, kurtosis) are highly sensitive to spalls, impacts, and impulsive transients, whereas spectral frequency-domain techniques (FFT, envelope analysis, spectral kurtosis) are required to detect distributed wear, rotational frequency shifts, and harmonic sidebands.
* **Relationship and Contrast:** This tutorial informs the feature selection in our study:
  - **Stage 1 (Fixed-Low):** Uses log-RMS, log-crest factor, and log-kurtosis to capture energy and impulsiveness.
  - **Stage 2 (Fixed-High):** Uses 16-band normalized energy and peak frequency to capture spectral envelope and harmonic deviations.
  Our contribution is translating this physical distinction into a mathematically bounded, two-stage adaptive decision selector with explicit operation accounting and proof of break-even escalation bounds.

### 2.4 Industrial Acoustic Benchmarks (ToyADMOS and MIMII)
* **Primary References:**
  1. Yuma Koizumi, Shoichiro Saito, Hisashi Uematsu, Noboru Harada, and Keisuke Imoto, "ToyADMOS: A Dataset of Miniature-Machine Operating Sounds for Anomalous Sound Detection," in *2019 IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA 2019)*, pp. 313–317, 2019.  
     DOI: [`10.1109/WASPAA.2019.8937164`](https://doi.org/10.1109/WASPAA.2019.8937164)
  2. Harsh Purohit et al., "MIMII Dataset: Sound Dataset for Malfunctioning Industrial Machine Investigation and Inspection," in *Proceedings of the Detection and Classification of Acoustic Scenes and Events 2019 Workshop (DCASE 2019)*, pp. 209–213, 2019.  
     arXiv: [`1909.09347`](https://arxiv.org/abs/1909.09347)
* **Core Contribution:** Standardized unsupervised anomalous sound detection (ASD) benchmarks across diverse machine families (pumps, valves, fans, slide rails), highlighting the impact of factory ambient noise and domain shifts on acoustic monitoring.
* **Relationship and Contrast:** DCASE Task 2 solutions typically evaluate heavy deep learning models (autoencoders, MobileNet, Conformer) operating on 128-bin Log-Mel spectrograms. In contrast, our study investigates whether a lightweight, two-stage classical signal-processing pipeline can operate with lower algorithmic proxy cost ($\approx 10,265$ to $29,250$ ops proxy) while maintaining detection quality across clean and noisy synthetic regimes. Real-world machine recordings from ToyADMOS/MIMII have not been evaluated in this repository; our signals remain strictly synthetic.

---

## 3. Comparison of Paradigms

| Dimension | Conventional Deep ASD (e.g. DCASE) | Classical Edge DSP (e.g. Fixed-High) | Our Adaptive Cascaded Approach |
| :--- | :--- | :--- | :--- |
| **Model Type** | Deep Autoencoder / MobileNet | 16-band rFFT Energy Distance | Two-Stage Dynamic Cascade |
| **Input Feature** | 128-band Log-Mel Spectrogram | 1024-point Hann-windowed rFFT | Time Moments + ZCF $\to$ Single-Pass rFFT |
| **Compute Cost** | $> 10^6$ MACs per second | 29,250 ops proxy per 1024-window | 10,265 ops proxy (Stage 1) / 39,515 ops proxy (Stage 2) |
| **Hardware Target** | Edge TPU / Cortex-A / Server | Targeted MCU platform (e.g. ESP32) | Targeted MCU platform (simulated ESP32) |
| **Early Exit?** | Rare (complex branch networks) | None (always computes FFT) | Yes (cheap decision for impacts & clean normals) |
| **Decision Logic** | Black-box neural reconstruction error | Robust z-score distance threshold | Hierarchical percentile & boundary checks |

---

## 4. Distinction Between Prior Art and Current Project Findings

1. **Prior Art Claims (Supported by Literature):**
   - Cascaded detection saves significant computation when the majority class can be identified by inexpensive features (Viola & Jones, 2001).
   - Time-domain moments effectively flag impulsive impacts, but are insensitive to tonal frequency shifts and subtle harmonics (Randall & Antoni, 2011).
   - Dynamic early exits reduce average compute cost across diverse workloads (Teerapittayanon et al., 2016).
   - Industrial machine monitoring exhibits heavy domain shifts and noise masking at low SNR (Koizumi et al., 2019; Purohit et al., 2019).

2. **Specific Findings of This Project (Derived Experimentally):**
   - **Theoretical Break-Even Proof:** In a two-stage cascade where Stage 1 costs $10,265$ ops and Stage 2 costs $29,250$ ops, the cascade costs *more* than standalone Fixed-High whenever the escalation rate exceeds **64.91%**. Adding even a $7N$ feature to Stage 1 compresses the break-even ceiling to **40.40%**.
   - **Order-of-Operations Sensitivity:** Evaluating zero-crossing frequency out-of-range *before* confident anomaly forces 100% of high-crest impact windows to escalate needlessly, turning a +28% savings into a -21.5% deficit.
   - **Single-Pass FFT Parity:** Unifying band-energy accumulation and peak-frequency estimation into a single rFFT pass removes redundant windowing and FFT execution in Python while matching embedded C++ firmware behavior.
   - **Subtle Harmonic Sensitivity Limit:** Candidate v1.1 ($P_{\text{normal}} = \text{p60}$) failed the pre-declared recall constraint on the held-out test set ($0.8750$ vs $\ge 0.8833$) due to 14 missed harmonic windows at high SNR. Candidate v1.2 ($P_{\text{normal}} = \text{p50}$, frozen under Protocol v1.2.0) resolves this limitation, achieving $0.9375$ recall (matching Fixed-High point estimate) on an independent held-out evaluation dataset.

---

## 5. Methodological and Experimental Safeguards

- **Synthetic Waveforms:** All signals analyzed in this repository are synthetically generated under controlled Python waveform definitions (stationary tones, second harmonics, exponential resonant impacts, additive white Gaussian noise). They do not represent real-world physical machine acoustic recordings.
- **Operation-Count Proxy:** Algorithmic operation counts ($10,265$, $29,250$, $39,515$) serve as an analytical work proxy. They do not constitute measured physical microcontroller power, energy dissipation, or battery longevity.
- **Desktop Runtime vs Embedded Latency:** Python CPU timing benchmarked with `perf_counter` reflects interpreter and NumPy overhead; it does not represent ESP32 execution latency.
- **Evaluation Independence:** Candidate v1.1 results on Phase 5 holdout (seed 20261013) and Candidate v1.2 results on independent holdout (seed 20261014) are evaluated on separate 20-group splits and are never combined or relabeled as a single test set.
- **Hardware Target Qualification:** The ESP32 is a simulated demonstration target; physical execution, real microphone I/O, and battery life have not been evaluated.
