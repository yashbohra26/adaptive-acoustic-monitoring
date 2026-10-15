# Literature matrix: adaptive acoustic monitoring

This matrix records the 21 PDF papers in `C:\Users\ASMIT\Downloads\ESA Papers\ESA Papers`. The full text of each PDF was extractable and reviewed for its abstract, method, reported evaluation, and discussion or conclusion. Bibliographic details below come from the article text, first-page citation block, journal headers, and PDF metadata where present. No external metadata lookup was used. “Author-reported” summarizes claims made in the paper; “relevance to this project” is our interpretation. In limitations fields, explicit author statements are attributed to the authors; comments about missing cost/hardware comparisons or narrow study scope are our reading of the supplied paper. Missing DOI data and source inconsistencies are called out rather than inferred.

## 1. Ji Wu, Fei Yang, and Wenkai Hu (2023)

**Verified reference:** “Unsupervised anomalous sound detection for industrial monitoring based on ArcFace classifier and Gaussian mixture model.” *Applied Acoustics*, 203, 109188 (2023). https://doi.org/10.1016/j.apacoust.2022.109188.

- **Problem and approach (authors):** Unsupervised detection of unknown machine anomalies. The authors train an ArcFace-based classifier to separate latent representations and use a Gaussian mixture model (GMM) to form anomaly-score boundaries.
- **Findings (authors):** They evaluate on DCASE 2020 Task 2 and the Case Western Reserve University bearing dataset. They report improved AUC and partial AUC against the compared methods.
- **Limitations/open questions (authors):** The conclusion says the classification-based method is less effective for non-stationary signals and identifies this as future work. The article says the data used are confidential.
- **Relevance to this project (our interpretation):** It is an unsupervised learned-feature comparator. Its latent classifier and GMM are more complex than our explicit time-domain/FFT cascade; the paper does not provide the same per-window operation accounting used here.
- **Verification:** Full text and bibliographic details verified from the PDF; DOI appears in the article.

## 2. Yawei Wang, Qiaoling Zhang, Weiwei Zhang, and Yi Zhang (2025)

**Verified reference:** “A lightweight framework for unsupervised anomalous sound detection based on selective learning of time-frequency domain features.” *Applied Acoustics*, 228, 110308 (2025). https://doi.org/10.1016/j.apacoust.2024.110308.

- **Problem and approach (authors):** Unsupervised ASD with selective use of time-frequency information. Spectrogram-frame selection (SFS) removes frames judged less informative; an autoencoder with a scaling gate (AEFS) selects frequency dimensions.
- **Findings (authors):** On DCASE 2020 Task 2, the authors report results above the unsupervised comparators and comparable to self-supervised methods. They report a model size of 0.08 MB.
- **Limitations/open questions (authors):** The method uses empirical machine-type attributes to classify sounds as stationary or non-stationary. The paper identifies this dependence as a limitation; its data statement says the data are confidential.
- **Relevance to this project (our interpretation):** Selective features are related to adaptive processing, but this paper does not evaluate our kind of explicit low-cost decision followed by conditional FFT work. Model size is not a measurement of energy or per-window operations.
- **Verification:** Full text and bibliographic details verified from the PDF and its metadata; DOI appears in both.

## 3. R. B. W. Heng and M. J. M. Nor (1998)

**Verified reference:** “Statistical Analysis of Sound and Vibration Signals for Monitoring Rolling Element Bearing Condition.” *Applied Acoustics*, 53(1–3), 211–226 (1998). No DOI was printed in the PDF or found in its PDF metadata.

- **Problem and approach (authors):** Bearing defect detection using sound-pressure and vibration signals. The study compares crest factor, kurtosis, skewness, and statistics derived from a beta distribution, implemented in a C++ program.
- **Findings (authors):** Under laboratory conditions, statistical features can distinguish defect types. Shaft speed affects the measures, particularly in the reported 1,500–2,500 rpm range. The authors find no significant advantage from beta-distribution parameters over kurtosis and crest factor.
- **Limitations/open questions (authors):** The paper describes its measurement setting as an ideal laboratory environment and explains that bearing-housing resonance affected results at some speeds.
- **Relevance to this project (our interpretation):** This is direct precedent for inexpensive statistical acoustic features and C++ implementation. It also shows why speed and rig dynamics can confound feature behavior.
- **Verification:** Full text and title, author, year, venue, volume, issue, and pages verified from the PDF. DOI is unavailable in the supplied source and should be checked manually if required by the final reference style.

## 4. Xiong Xu, He Wen, HaiJun Lin, ZiXin Li, and Cong Huang (2024)

**Verified reference:** “Online detection method for variable load conditions and anomalous sound of hydro turbines using correlation analysis and PCA-adaptive-K-means.” *Measurement*, 224, 113846 (2024). https://doi.org/10.1016/j.measurement.2023.113846.

- **Problem and approach (authors):** Online sound monitoring of hydro turbines must separate load changes from anomalies. The method uses online correlation analysis for load-state detection, then time-frequency and wavelet-packet features, PCA, and adaptive K-means under stable load.
- **Findings (authors):** The authors report tests on measurements from a hydropower station and detection rates of 98.3% for anomalous sound and 96.5% for variable load in their table, exceeding the listed comparison methods on those measures. They describe one-second online updates.
- **Limitations/open questions (authors):** The authors state that they lack permission to share the data and propose a larger hydro-turbine sound dataset as future work.
- **Relevance to this project (our interpretation):** It provides real-machine evidence that operating context matters. Its adaptive K-means responds to load variation; it is not a staged low-cost/high-cost signal-processing cascade, and the paper does not report a comparable operation-count proxy.
- **Verification:** Full text, metadata, citation details, and DOI verified from the PDF.

## 5. Shuyue Zhang, Jingxin Zhang, Shuo Shan, Yu Shen, Chenxi Li, Sheng Zhu, Kanjian Zhang, and Haikun Wei (2025)

**Verified reference:** “Diesel generator set anomalous sound detection based on lightweight convolutional autoencoder and prior knowledge integration.” *Measurement*, 243, 116392 (2025). https://doi.org/10.1016/j.measurement.2024.116392.

- **Problem and approach (authors):** Unsupervised fault detection in diesel-generator sounds. A lightweight convolutional autoencoder uses Mel-spectrogram reconstruction and Mahalanobis distance, with prior knowledge represented through spectral kurtosis.
- **Findings (authors):** The authors evaluate recordings from a diesel generator operating at different loads in an Antarctic simulated power-generation cabin and report improved fault sensitivity from adding prior spectral information.
- **Limitations/open questions (authors):** The article says testing did not cover all load conditions or faults and focused on three fault types. It also says Antarctic temperature and pressure effects were not yet modeled.
- **Relevance to this project (our interpretation):** This study grounds acoustic monitoring in a machine-specific recording setting and makes explicit the gap between synthetic test cases and diverse operating conditions. Its autoencoder is not a matched baseline for our cost proxy.
- **Verification:** Full text, metadata, citation details, and DOI verified from the PDF.

## 6. Emin Germen, Murat Başaran, and Mehmet Fidan (2014)

**Verified reference:** “Sound based induction motor fault diagnosis using Kohonen self-organizing map.” *Mechanical Systems and Signal Processing*, 46, 45–58 (2014). No DOI was printed in the PDF or found in its PDF metadata.

- **Problem and approach (authors):** Classify mechanical and electrical faults in squirrel-cage induction motors from acoustic recordings. Five microphones arranged around a motor provide signals for correlation and wavelet-based features; Kohonen self-organizing maps (SOM) and learning-vector quantization (LVQ) classify fault types.
- **Findings (authors):** The paper reports separation of healthy and faulty motors and classification of several fault types. It notes that acoustic features are more sensitive to environmental interference than current or vibration analysis.
- **Limitations/open questions (authors):** The study uses a specific test bed and five-microphone arrangement; the authors discuss environmental noise as a challenge.
- **Relevance to this project (our interpretation):** It shows a classical feature-based acoustic diagnosis pipeline, but with a multi-microphone setup and fault-type classification rather than our single-window anomaly decision and staged work model.
- **Verification:** Full text and citation fields verified from the PDF, including the journal header on the article pages. DOI is not present in the supplied source and needs manual lookup if required.

## 7. Emanuele Di Fiore, Antonino Ferraro, Antonio Galli, Vincenzo Moscato, and Giancarlo Sperlì (2022)

**Verified reference:** “An anomalous sound detection methodology for predictive maintenance.” *Expert Systems with Applications*, 209, 118324 (2022). https://doi.org/10.1016/j.eswa.2022.118324.

- **Problem and approach (authors):** ASD for multiple instances of industrial machine types. The authors condition an autoencoder on equipment identity, using Mel-spectrograms and machine identifiers; they test convolutional and LSTM variants.
- **Findings (authors):** On DCASE 2020 Task 2, the paper reports improvements of up to 17.61% over corresponding unconditioned autoencoders and the baseline, especially for partial AUC. It also discusses inference time and memory relative to other approaches.
- **Limitations/open questions (authors):** The conclusion proposes exploring other conditioning networks and preprocessing strategies. The approach relies on machine-identity information, which is an additional input unavailable to our selector.
- **Relevance to this project (our interpretation):** It is a learned, identity-conditioned comparator and illustrates a different use of context from our signal-only selector. Its compute measures are not directly comparable with our arithmetic proxy.
- **Verification:** Full text, metadata, citation details, and DOI verified from the PDF.

## 8. Jisheng Bai, Jianfeng Chen, Mou Wang, Muhammad Saad Ayub, and Qingli Yan (2023)

**Verified reference:** “SSDPT: Self-supervised dual-path transformer for anomalous sound detection.” *Digital Signal Processing*, 135, 103939 (2023). https://doi.org/10.1016/j.dsp.2023.103939.

- **Problem and approach (authors):** Detect machine anomalies when fault sounds are scarce. The self-supervised dual-path Transformer models temporal and frequency components of segmented log-Mel features with attention blocks; masked reconstruction and machine-ID classification are used as learning objectives.
- **Findings (authors):** On DCASE 2021 Task 2, the authors report that their system improves harmonic AUC relative to the methods in their comparison. Their ensemble reports h-AUC 0.739 and h-pAUC 0.626.
- **Limitations/open questions (authors):** The article focuses on benchmark metrics and does not provide a matched microcontroller energy or operation-count evaluation.
- **Relevance to this project (our interpretation):** It represents a high-capacity learned approach to temporal and spectral modeling, in contrast to the transparent, threshold-based cascade here.
- **Verification:** Full text and bibliographic details, including DOI, verified from the PDF.

## 9. Shengbing Chen, Yong Sun, Junjie Wang, Mengyuan Wan, Mengyuan Liu, and Xiaofan Li (2025)

**Verified reference:** “A multi-scale dual-decoder autoencoder model for domain-shift machine sound anomaly detection.” *Digital Signal Processing*, 156, 104813 (2025). https://doi.org/10.1016/j.dsp.2024.104813.

- **Problem and approach (authors):** Unsupervised anomaly detection under changes in operating conditions and environmental noise. The MS-D2AE combines a multi-scale feature-fusion module, two decoders, and feature-domain mixing.
- **Findings (authors):** The paper evaluates on DCASE 2022 and 2023 Task 2 and reports improvements on most machine types in its comparisons.
- **Limitations/open questions (authors):** The authors state that results remain unsatisfactory for a small number of machine types and propose improving accuracy and reducing false positives.
- **Relevance to this project (our interpretation):** It emphasizes domain shift as a central challenge. Its neural domain-mixing approach differs from our controlled-SNR synthetic study, which does not test machine or environment shift.
- **Verification:** Full text, metadata, citation details, and DOI verified from the PDF.

## 10. Yi Zhang, Jie Feng, Qiaoling Zhang, Junyao Hu, and Weiwei Zhang (2025)

**Verified reference:** “An unsupervised anomalous sound detection method based on similarity-driven automatic feature selection.” *Digital Signal Processing*, 161, 105123 (2025). https://doi.org/10.1016/j.dsp.2025.105123.

- **Problem and approach (authors):** Select features that fit the temporal characteristics of machine sounds. A similarity-based classifier groups sound types, selects spectrogram frames, and feeds them to a beta-VAE for anomaly detection.
- **Findings (authors):** On DCASE 2020 and 2022 Task 2, the authors report results above the unsupervised models in their comparison and a model size of 0.12 MB.
- **Limitations/open questions (authors):** The authors say the method does not stably learn sound characteristics under domain shift and identify this as future work.
- **Relevance to this project (our interpretation):** Like the project, it selects which acoustic information receives further processing. Its selection is learned feature/frame selection, not an explicit cascade with a per-window stage cost.
- **Verification:** Full text, metadata, citation details, and DOI verified from the PDF.

## 11. Christof Pichler, Markus Neumayer, Bernhard Schweighofer, Christoph Feilmayr, Stefan Schuster, and Hannes Wegleiter (2023)

**Verified reference:** “Acoustic-Based Detection Technique for Identifying Worn-Out Components in Large-Scale Industrial Machinery.” *IEEE Sensors Letters*, 7(9), article 6006204 (2023). DOI in PDF metadata: https://doi.org/10.1109/LSENS.2023.3307103.

- **Problem and approach (authors):** Detect worn components in a large industrial machine hall. The study compares conventional sound features with physically motivated features, trains a GMM on normal data, and analyzes feature performance and window length.
- **Findings (authors):** In the study's application, the authors report that MFCCs perform best, with physically motivated features more interpretable and easier to compute. They state that combining additional feature types did not notably improve overall performance.
- **Limitations/open questions (authors):** The conclusion calls the system a starting point and notes a notable false-positive rate. The authors propose using continuous failure sound and temporal information.
- **Relevance to this project (our interpretation):** It supports comparing interpretable low-cost features with richer audio representations and highlights the importance of false alarms. It does not evaluate a conditional two-stage policy.
- **Verification:** Full text and article-page citation information verified. DOI was verified from the PDF metadata subject field; it is not clearly printed in the extracted article text.

## 12. Yasuji Ota and Masashi Unoki (2023)

**Verified reference:** “Anomalous Sound Detection for Industrial Machines Using Acoustical Features Related to Timbral Metrics.” *IEEE Access*, 11 (2023). https://doi.org/10.1109/ACCESS.2023.3294334.

- **Problem and approach (authors):** Detect industrial machine anomalies using sound characteristics associated with human timbre perception. The TF-ASD method combines five timbral metrics with two short-term features and an SVM.
- **Findings (authors):** On MIMII, the paper reports average classification accuracy of 0.984 over 16 machine types and models. The authors analyze feature contributions and report that timbral and short-term features are useful in their setting.
- **Limitations/open questions (authors):** They identify further feature development, time-series analysis, and study of the relation between timbre and human perception as future work.
- **Relevance to this project (our interpretation):** It provides an interpretable feature-based comparator on recorded machine sounds. Its SVM classification accuracy is not directly comparable to the project's held-out synthetic F1/recall metrics.
- **Verification:** Full text, title-page metadata, publication year, venue, and DOI verified from the PDF.

## 13. Yaoguang Wang, Yaohao Zheng, Yunxiang Zhang, Yongsheng Xie, Sen Xu, Ying Hu, and Liang He (2021)

**Verified reference:** “Unsupervised Anomalous Sound Detection for Machine Condition Monitoring Using Classification-Based Methods.” *Applied Sciences*, 11(23), 11128 (2021). https://doi.org/10.3390/app112311128.

- **Problem and approach (authors):** Improve unsupervised ASD by using an outlier classifier and an ID classifier that scores classification confidence and hidden-embedding similarity. The training process uses samples from other machine IDs.
- **Findings (authors):** On DCASE 2020 Task 2, the ensemble reports average AUC 95.82% and pAUC 92.32%. The authors state that it does not perform best on every machine type.
- **Limitations/open questions (authors):** The paper identifies generalization to additional machine types as future work. Its use of other machine-ID data is an assumption that may not hold in every deployment.
- **Relevance to this project (our interpretation):** It shows how cross-machine data can improve learned decision boundaries, while our selector uses only each signal window and development calibration boundaries.
- **Verification:** Full text, citation block, PDF metadata, DOI, and reported abstract/results verified.

## 14. Siddharth Sigtia, Adam M. Stark, Sacha Krstulović, and Mark D. Plumbley (2016)

**Verified reference:** “Automatic Environmental Sound Recognition: Performance Versus Computational Cost.” *IEEE/ACM Transactions on Audio, Speech, and Language Processing*, 24(11), 2096–2107 (2016). DOI in PDF metadata: https://doi.org/10.1109/TASLP.2016.2592698.

- **Problem and approach (authors):** Compare environmental sound-recognition accuracy as a function of computational cost for Gaussian mixture models, support-vector machines, and deep neural networks.
- **Findings (authors):** The paper reports that DNNs attain the strongest recognition-performance-to-cost results over the tested range; GMMs provide a lower-cost baseline and SVMs fall between them.
- **Limitations/open questions (authors):** The authors note the scarcity of large datasets and propose studying memory efficiency. The task is environmental sound recognition, not machine anomaly detection.
- **Relevance to this project (our interpretation):** It motivates reporting a cost-quality trade-off. Its model/runtime cost analysis and recognition task are not directly comparable to our arithmetic operation proxy or anomaly metrics.
- **Verification:** Full text and title-page citation information verified. DOI was verified in the PDF metadata subject field.

## 15. Yuki Tagawa, Rytis Maskeliūnas, and Robertas Damaševičius (2021)

**Verified reference:** “Acoustic Anomaly Detection of Mechanical Failures in Noisy Real-Life Factory Environments.” *Electronics*, 10(19), 2329 (2021). https://doi.org/10.3390/electronics10192329.

- **Problem and approach (authors):** Detect mechanical anomalies in noisy factory sound without dedicated sensors per machine. A GAN reconstructs acoustic signals; the study also evaluates an unscented Kalman filter and an MSE loss with L2 regularization.
- **Findings (authors):** On MIMII, the authors report advantages over OC-SVM and autoencoder-decoder baselines, especially for noisy pump data under the proposed configuration.
- **Limitations/open questions (authors):** They propose improving low-SNR feature handling and examining short-time models and alternative loss functions.
- **Relevance to this project (our interpretation):** It evaluates real recorded sounds under noise, unlike our synthetic SNR conditions. Its learned reconstruction cost is not comparable to our operation-count model.
- **Verification:** Full text, citation block, metadata, and DOI verified from the PDF.

## 16. Iordanis Thoidis, Marios Giouvanakis, and George Papanikolaou (2021)

**Verified reference:** “Semi-Supervised Machine Condition Monitoring by Learning Deep Discriminative Audio Features.” *Electronics*, 10(20), 2471 (2021). https://doi.org/10.3390/electronics10202471.

- **Problem and approach (authors):** Learn audio representations for machines and detect abnormal operation. Raw multichannel audio is processed by a temporal CNN; supervised representation learning is combined with a one-class detector, and a front end exploits spatial information from circular microphone arrays.
- **Findings (authors):** On MIMII, the authors report a mean AUC of 91.0% and improved results from multichannel and spatially invariant features.
- **Limitations/open questions (authors):** The method assumes a semi-supervised training setup and multichannel audio options; the paper does not report physical energy measurements.
- **Relevance to this project (our interpretation):** It illustrates the performance available from learned spatial features and the additional input/model assumptions beyond our single-window, single-signal selector.
- **Verification:** Full text, citation block, PDF metadata, DOI, and reported abstract/results verified.

## 17. Gbanaibolou Jombo and Yu Zhang (2023)

**Verified reference:** “Acoustic-Based Machine Condition Monitoring—Methods and Challenges.” *Eng*, 4, 47–79 (2023). https://doi.org/10.3390/eng4010004.

- **Problem and approach (authors):** Review acoustic machine-condition monitoring methods, available datasets, and barriers to practical use.
- **Findings (authors):** The review discusses the advantages of non-contact acoustic sensors and recurring sensitivity to background noise and operating-condition changes. It surveys signal-processing and learning-based approaches.
- **Limitations/open questions (authors):** The article is a review, not a matched experimental comparison. It identifies unresolved questions in practical analysis, including noisy, multichannel settings.
- **Relevance to this project (our interpretation):** It provides a broad map of application and method issues, but it cannot validate this project's detector or operation proxy.
- **Verification:** Full text, title-page citation block, metadata, DOI, volume, and pages verified.

## 18. Nishchal K. Verma, Rahul Kumar Sevakula, Sonal Dixit, and Al Salour (2016)

**Verified reference:** “Intelligent Condition Based Monitoring Using Acoustic Signals for Air Compressors.” *IEEE Transactions on Reliability*, 65(1), 291–309 (2016). DOI in PDF metadata: https://doi.org/10.1109/TR.2015.2459684.

- **Problem and approach (authors):** Build an acoustic fault-diagnosis process for reciprocating air compressors, including acquisition, sensor-position analysis, preprocessing, feature extraction/selection, and multiclass classification.
- **Findings (authors):** The authors report a real-time system for eight compressor states (one healthy and seven faulty) that detected all faults from recordings at one sensor position.
- **Limitations/open questions (authors):** The paper is specific to a compressor and its designated states. The reviewed source does not establish performance across other machines or give a cost measure comparable to this project's operation proxy.
- **Relevance to this project (our interpretation):** It is a real-machine acoustic deployment precedent; the project's synthetic comparison is narrower and has no physical device validation.
- **Verification:** Full text and title-page citation information verified. DOI was verified in PDF metadata.

## 19. Hoang Van Truong, Nguyen Chi Hieu, Pham Ngoc Giao, and Nguyen Xuan Phong (2021)

**Verified reference:** “Unsupervised Detection of Anomalous Sound for Machine Condition Monitoring using Fully Connected U-Net.” *Journal of ICT Research and Applications*, 15(1), 41–55 (2021). https://doi.org/10.5614/itbj.ict.res.appl.2021.15.1.3.

- **Problem and approach (authors):** Improve reconstruction-based ASD for non-stationary sounds. The authors introduce mixed acoustic features and a fully connected U-Net autoencoder.
- **Findings (authors):** On the DCASE baseline dataset, the paper reports mean AUC 83.38% and pAUC 64.51%, over the stated baseline by 13.43 and 8.13 percentage points, respectively.
- **Limitations/open questions (authors):** The conclusion proposes data augmentation and further work on complex sounds and real environments. It does not report hardware or energy measurements.
- **Relevance to this project (our interpretation):** It reinforces that feature representation and non-stationarity affect anomaly detection, but uses a learned reconstruction model rather than explicit conditional stages.
- **Verification:** Article title page supplies the full citation and DOI. The PDF's internal title metadata is a template placeholder, so its metadata field is not valid; the article-page citation was used instead. Full text was readable.

## 20. Ahmad Qurthobi, Rytis Maskeliūnas, and Robertas Damaševičius (2022)

**Verified reference:** “Detection of Mechanical Failures in Industrial Machines Using Overlapping Acoustic Anomalies: A Systematic Literature Review.” *Sensors*, 22(10), 3888 (2022). https://doi.org/10.3390/s22103888.

- **Problem and approach (authors):** A PRISMA-based review of acoustic mechanical-failure detection, with searches in IEEE Xplore, ScienceDirect, and SpringerLink.
- **Findings (authors):** The abstract reports that acoustic-emission studies dominate and that noisy-condition failure detection remains limited. The authors discuss wear, cracks, seeded faults, and machine-learning methods.
- **Limitations/open questions (authors):** The paper's abstract says 52 articles were selected, while its conclusion says 53 studies met inclusion criteria. This internal count discrepancy should be checked against the review's study-selection flow before citing the exact count.
- **Relevance to this project (our interpretation):** It supports noise robustness and validation conditions as open concerns, while also providing a systematic review that should inform a fuller review beyond the current paper folder.
- **Verification:** Full text, citation block, metadata, DOI, title, and publication information verified. The selected-study count remains internally inconsistent and is not repeated as a settled number here.

## 21. Yuma Koizumi, Shoichiro Saito, Hisashi Uematsu, Yuta Kawachi, and Noboru Harada (2019)

**Verified reference:** “Unsupervised Detection of Anomalous Sound Based on Deep Learning and the Neyman–Pearson Lemma.” *IEEE/ACM Transactions on Audio, Speech, and Language Processing*, 27(1), 212–223 (2019). DOI in PDF metadata: https://doi.org/10.1109/TASLP.2018.2877258.

- **Problem and approach (authors):** Train an autoencoder objective to maximize true-positive rate under a specified low false-positive-rate constraint. The method uses rejection sampling to construct surrogate anomalies from the complement of normal data.
- **Findings (authors):** Experiments on synthetic data report improved detection under low-FPR settings, and the authors also report detection in real environments.
- **Limitations/open questions (authors):** The discussion cautions that using only a subset of anomalous sounds to calculate the objective may not guarantee detection of unknown anomalies. It also notes that machine- or context-specific alert post-processing remains open.
- **Relevance to this project (our interpretation):** It links anomaly-detection thresholds to explicit operating constraints. It does not supply the project's comparison rule; that rule remains a decision for the research team.
- **Verification:** Full text and citation details verified. DOI was verified in PDF metadata; the article pages begin at 212 and end at 223.

## Review-wide observations

- The folder includes classical statistical features, learned unsupervised and semi-supervised ASD, machine-specific field or laboratory studies, two reviews, and one environmental sound recognition paper with explicit computational-cost comparisons.
- Only a subset reports a cost-related quantity, and those quantities differ: model size, inference time, or a general cost/performance comparison. None is directly interchangeable with this project's arithmetic operation-count proxy.
- Several sources test recorded machine audio or operational conditions, while this project currently tests controlled synthetic windows. The reported metrics and splits also differ, so their numerical results should not be compared directly with Phase 5 or v1.2.
- A source-based research gap is the lack of a directly matched comparison in this reviewed set for a transparent time-domain first stage, conditional spectral escalation, fixed-low/fixed-high references, and per-window work accounting under a grouped held-out protocol. This is a gap in the papers supplied and reviewed, not a claim that no such work exists.

## Manual verification flags

1. Papers 3 and 6 contain no DOI in the supplied PDF or its metadata. Their title, authors, year, venue, and pages are transcribed from the papers; locate and verify a DOI if the intended reference style requires one.
2. Paper 19's internal PDF metadata has a template title and should not be used as citation metadata. The citation on its first page was used.
3. Paper 20 reports different study counts in the abstract and conclusion (52 and 53). Confirm the number from the article's PRISMA diagram before quoting it.
4. The abstract/conclusion of paper 12 reports “accuracy” for a classification setup. Do not treat that value as anomaly-detection recall or compare it directly with this project's F1/recall.
