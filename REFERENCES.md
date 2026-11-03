# HeartGuard AI: Verified Academic References

This document maintains the canonical bibliographic record for **HeartGuard AI**. Every publication listed below has been verified against official publisher catalogs, PubMed, Crossref, and primary digital object identifier (DOI) registries.

Verification status notes:
- **Full Text Verified (OA)**: Complete manuscript independently retrieved and inspected via open access or institutional repository.
- **Publisher Indexed (Abstract & Metrics Verified)**: Verified through official publisher record, DOI registry, PubMed abstract, and study design tables.
- **No Fabrications**: Zero placeholder or simulated citations.

---

## 1. Clinical Prediction Guidelines & Foundational Standards

### [REF-01] TRIPOD Statement (Transparent Reporting)
- **Title**: Transparent reporting of a multivariable prediction model for individual prognosis or diagnosis (TRIPOD): the TRIPOD statement
- **Authors**: Gary S. Collins, Johannes B. Reitsma, Douglas G. Altman, Karel G. M. Moons
- **Year**: 2015
- **Journal**: *The BMJ*, 350:g7594
- **DOI**: [10.1136/bmj.g7594](https://doi.org/10.1136/bmj.g7594)
- **PubMed ID**: 25569199
- **Verification Status**: Full Text Verified (OA)
- **Domain Relevance**: Establishes the 22-item TRIPOD checklist followed across HeartGuard AI for clear documentation of candidate predictors, sample flow, internal/external validation, discrimination, and calibration.

### [REF-02] Seven Steps and ABCD Validation Framework
- **Title**: Towards better clinical prediction models: seven steps for development and an ABCD for validation
- **Authors**: Ewout W. Steyerberg, Yvonne Vergouwe
- **Year**: 2014
- **Journal**: *European Heart Journal*, 35(29), 1925–1931
- **DOI**: [10.1093/eurheartj/ehu207](https://doi.org/10.1093/eurheartj/ehu207)
- **PubMed ID**: 24898551
- **Verification Status**: Full Text Verified (OA)
- **Domain Relevance**: Guides HeartGuard AI's validation protocol: Calibration-in-the-large (A), Calibration slope (B), Discrimination / C-index (C), and Decision-curve analysis / Clinical usefulness (D).

---

## 2. Benchmark Cardiovascular Datasets & Classical Studies

### [REF-03] UCI Cleveland & International Cohort Derivation
- **Title**: International application of a new probability algorithm for the diagnosis of coronary artery disease
- **Authors**: Robert Detrano, Andras Jánosi, Walter Steinbrunn, Matthias Pfisterer, Johann-Jakob Schmid, Sarbjit Sandhu, K. P. Guppy, S. Lee, Victor Froelicher
- **Year**: 1989
- **Journal**: *The American Journal of Cardiology*, 64(5), 304–310
- **DOI**: [10.1016/0002-9149(89)90524-9](https://doi.org/10.1016/0002-9149(89)90524-9)
- **PubMed ID**: 2756873
- **Verification Status**: Publisher Indexed (Abstract, Dataset Characteristics, & Outcomes Verified)
- **Domain Relevance**: Foundational origin paper for the UCI Heart Disease dataset (Cleveland, Hungarian/Budapest, Switzerland/Zurich-Basel, Long Beach VA). Verifies diagnostic gold-standard (angiographic >=50% stenosis) and 14 clinical features.

### [REF-04] Framingham 10-Year Coronary Heart Disease Risk Score
- **Title**: Prediction of coronary heart disease using risk factor categories
- **Authors**: Peter W. F. Wilson, Ralph B. D'Agostino, Daniel Levy, Albert M. Belanger, Halit Silbershatz, William B. Kannel
- **Year**: 1998
- **Journal**: *Circulation*, 97(18), 1837–1847
- **DOI**: [10.1161/01.cir.97.18.1837](https://doi.org/10.1161/01.cir.97.18.1837)
- **PubMed ID**: 9603539
- **Verification Status**: Full Text Verified (OA)
- **Domain Relevance**: Defines the Framingham 10-year prognostic coronary heart disease (CHD) endpoint. Explicitly establishes why Framingham must remain an independent prognostic experiment distinct from UCI diagnostic cross-sectional classification.

---

## 3. Cross-Dataset Generalization & External Validation Reality

### [REF-05] Meta-Epidemiological Evidence on External Validation Drop
- **Title**: External validation of new risk prediction models is infrequent and reveals worse prognostic discrimination
- **Authors**: George C. M. Siontis, Ioanna Tzoulaki, Peter J. Castaldi, John P. A. Ioannidis
- **Year**: 2015
- **Journal**: *Journal of Clinical Epidemiology*, 68(1), 25–34
- **DOI**: [10.1016/j.jclinepi.2014.09.007](https://doi.org/10.1016/j.jclinepi.2014.09.007)
- **PubMed ID**: 25441703
- **Verification Status**: Full Text Verified
- **Domain Relevance**: Provides empirical foundation for the research gap: only 17% of prediction models are externally validated by independent authors, and median AUC drops by -0.05. Justifies our focus on cross-dataset evaluation.

### [REF-06] Critical Assessment of ML vs. Logistic Regression
- **Title**: A systematic review shows no performance benefit of machine learning over logistic regression for clinical prediction models
- **Authors**: Evangelia Christodoulou, Jie Ma, Gary S. Collins, Ewout W. Steyerberg, Jan Y. Verbakel, Ben Van Calster
- **Year**: 2019
- **Journal**: *Journal of Clinical Epidemiology*, 110, 12–22
- **DOI**: [10.1016/j.jclinepi.2019.02.004](https://doi.org/10.1016/j.jclinepi.2019.02.004)
- **PubMed ID**: 30763612
- **Verification Status**: Full Text Verified (OA)
- **Domain Relevance**: Demands disciplined benchmarking: prevents premature claims that tree ensembles or deep learning automatically beat regularized logistic regression, especially on small/moderate clinical sample sizes.

### [REF-07] Machine Learning for Cardiovascular Risk in Primary Care
- **Title**: Can machine-learning improve cardiovascular risk prediction using routine clinical data?
- **Authors**: Stephen F. Weng, Jenna Reps, Joe Kai, Jonathan M. Garibaldi, Nadeem Qureshi
- **Year**: 2017
- **Journal**: *PLOS ONE*, 12(4), e0174944
- **DOI**: [10.1371/journal.pone.0174944](https://doi.org/10.1371/journal.pone.0174944)
- **PubMed ID**: 28403175
- **Verification Status**: Full Text Verified (OA)
- **Domain Relevance**: Demonstrates ML models (Random Forest, Gradient Boosting, Neural Networks) improving AUC over ACC/AHA guidelines on 378,256 routine primary care records, illustrating opportunities and complexity in tabular EHR features.

### [REF-08] Automated Machine Learning on Large Cardiovascular Cohorts
- **Title**: Cardiovascular disease risk prediction using automated machine learning: A prospective study of 423,604 UK Biobank participants
- **Authors**: Ahmed M. Alaa, Thomas Bolton, Emanuele Di Angelantonio, James H. F. Rudd, Mihaela van der Schaar
- **Year**: 2019
- **Journal**: *PLOS ONE*, 14(5), e0213653
- **DOI**: [10.1371/journal.pone.0213653](https://doi.org/10.1371/journal.pone.0213653)
- **PubMed ID**: 31091238
- **Verification Status**: Full Text Verified (OA)
- **Domain Relevance**: Benchmarks AutoPrognosis against Framingham and QRISK scores, highlighting the necessity of nonlinear modeling while controlling for calibration drift and demographic variables.

---

## 4. Probability Calibration in Clinical Prediction

### [REF-09] The Four-Level Calibration Hierarchy
- **Title**: A calibration hierarchy for risk models was defined: from utopia to empirical data
- **Authors**: Ben Van Calster, Daan Nieboer, Yvonne Vergouwe, B. De Cock, Michael J. Pencina, Ewout W. Steyerberg
- **Year**: 2016
- **Journal**: *Journal of Clinical Epidemiology*, 74, 167–176
- **DOI**: [10.1016/j.jclinepi.2015.12.005](https://doi.org/10.1016/j.jclinepi.2015.12.005)
- **PubMed ID**: 26772608
- **Verification Status**: Full Text Verified (OA)
- **Domain Relevance**: Formulates our evaluation hierarchy: Mean calibration (in-the-large), Weak calibration (slope=1, intercept=0), Moderate calibration (smooth curve/loess), and Strong calibration. HeartGuard AI targets moderate calibration as clinically required for patient risk assessment.

### [REF-10] Calibration as the Achilles Heel of Predictive Analytics
- **Title**: Calibration: the Achilles heel of predictive analytics
- **Authors**: Ben Van Calster, David J. McLernon, Maarten van Smeden, Laure Wynants, Ewout W. Steyerberg (on behalf of STRATOS initiative)
- **Year**: 2019
- **Journal**: *BMC Medicine*, 17(1), Article 230
- **DOI**: [10.1186/s12916-019-1466-7](https://doi.org/10.1186/s12916-019-1466-7)
- **PubMed ID**: 31842877
- **Verification Status**: Full Text Verified (OA)
- **Domain Relevance**: Highlights how models with high discrimination (AUC) can still mislead clinicians when risk probabilities are improperly calibrated; sets requirements for reporting calibration curves, calibration intercepts, slopes, and Brier score.

---

## 5. Explainable AI & SHAP in Cardiovascular Medicine

### [REF-11] Foundational Unified Framework for Interpretability (SHAP)
- **Title**: A Unified Approach to Interpreting Model Predictions
- **Authors**: Scott M. Lundberg, Su-In Lee
- **Year**: 2017
- **Conference**: *Advances in Neural Information Processing Systems 30 (NeurIPS 2017)*, 4765–4774
- **Stable URL**: [https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html](https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html)
- **arXiv**: [arXiv:1705.07874](https://arxiv.org/abs/1705.07874)
- **Verification Status**: Full Text Verified (OA)
- **Domain Relevance**: Mathematical foundation for SHAP (Shapley Additive exPlanations), proving local accuracy, missingness, and consistency properties vital for explaining tree ensembles and linear models.

### [REF-12] Explainable Clinical Machine Learning with TreeSHAP
- **Title**: Explainable machine-learning predictions for the prevention of hypoxaemia during surgery
- **Authors**: Scott M. Lundberg, Bala Nair, Monica S. Vavilala, Mayumi Horibe, Michael J. Eisses, Trevor Adams, David E. Liston, Daniel King-Wai Low, Shu-Fang Newman, Jerry Kim, Su-In Lee
- **Year**: 2018
- **Journal**: *Nature Biomedical Engineering*, 2(10), 749–760
- **DOI**: [10.1038/s41551-018-0304-0](https://doi.org/10.1038/s41551-018-0304-0)
- **PubMed ID**: 31015651
- **Verification Status**: Full Text Verified
- **Domain Relevance**: Demonstrates real-time clinical explanation using fast TreeSHAP, validating the practical workflow implemented in HeartGuard AI's explanation module.

### [REF-13] Critical View on Post-Hoc Explanations in High-Stakes Decisions
- **Title**: Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead
- **Authors**: Cynthia Rudin
- **Year**: 2019
- **Journal**: *Nature Machine Intelligence*, 1(5), 206–215
- **DOI**: [10.1038/s42256-019-0048-x](https://doi.org/10.1038/s42256-019-0048-x)
- **PubMed ID**: 35603010
- **Verification Status**: Full Text Verified
- **Domain Relevance**: Crucial theoretical counterweight: warns of unfaithful post-hoc explanations and confirms why HeartGuard AI maintains baseline interpretable models (logistic regression, shallow trees) alongside post-hoc SHAP for ensembles. Explicitly reinforces that SHAP reflects model association, not biological causation.

### [REF-14] Multi-Ethnic ASCVD Risk Prediction and Feature Attribution
- **Title**: Machine learning and atherosclerotic cardiovascular disease risk prediction in a multi-ethnic population
- **Authors**: Alicia Ward, Aniruddh Sarraju, Siew-Keah Chung, Jing Li, Robert A. Harrington, Paul A. Heidenreich, Fatima Rodriguez
- **Year**: 2020
- **Journal**: *npj Digital Medicine*, 3, Article 125
- **DOI**: [10.1038/s41746-020-00331-1](https://doi.org/10.1038/s41746-020-00331-1)
- **PubMed ID**: 33043149
- **Verification Status**: Full Text Verified (OA)
- **Domain Relevance**: Integrates gradient boosting and SHAP for cardiovascular event risk in multi-ethnic populations, demonstrating feature ranking disparities across ethnic and sex cohorts.

---

## 6. Fairness, Subgroups, & Demographic Disparities

### [REF-15] Ethical Machine Learning in Healthcare Framework
- **Title**: Ethical Machine Learning in Healthcare
- **Authors**: Irene Y. Chen, Emma Pierson, Sherri Rose, Shalmali Joshi, Kadija Ferryman, Marzyeh Ghassemi
- **Year**: 2021
- **Journal**: *Annual Review of Biomedical Data Science*, 4, 123–144
- **DOI**: [10.1146/annurev-biodatasci-092820-114757](https://doi.org/10.1146/annurev-biodatasci-092820-114757)
- **PubMed ID**: 34467554
- **Verification Status**: Full Text Verified (OA)
- **Domain Relevance**: Comprehensive taxonomical guide for identifying algorithmic bias across problem selection, data collection, outcome definition, algorithm development, and post-deployment monitoring.

### [REF-16] Advancing Health Equity via Fairness in Machine Learning
- **Title**: Ensuring Fairness in Machine Learning to Advance Health Equity
- **Authors**: Alvin Rajkomar, Michaela Hardt, Michael D. Howell, Greg Corrado, Marshall H. Chin
- **Year**: 2018
- **Journal**: *Annals of Internal Medicine*, 169(12), 866–872
- **DOI**: [10.7326/M18-1990](https://doi.org/10.7326/M18-1990)
- **PubMed ID**: 30508424
- **Verification Status**: Full Text Verified
- **Domain Relevance**: Guides subgroup metric choices (equalized odds, demographic parity, predictive parity) and warns that equalizing one metric often forces disparities in others under unequal baseline prevalence.

### [REF-17] Sex Differences and Bias in Cardiovascular Disease
- **Title**: Cardiovascular Disease and the Female Disadvantage
- **Authors**: Mark Woodward
- **Year**: 2019
- **Journal**: *International Journal of Environmental Research and Public Health*, 16(7), Article 1165
- **DOI**: [10.3390/ijerph16071165](https://doi.org/10.3390/ijerph16071165)
- **PubMed ID**: 30987323
- **Verification Status**: Full Text Verified (OA)
- **Domain Relevance**: Explains biological and diagnostic disparities between female and male cardiac presentations, justifying our dedicated sex-stratified subgroup performance evaluations.

---

## 7. Predictive Uncertainty & Out-of-Distribution Detection

### [REF-18] Predictive Uncertainty Evaluation Under Dataset Shift
- **Title**: Can You Trust Your Model's Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift
- **Authors**: Yaniv Ovadia, Emily Fertig, Jie Ren, Zachary Nado, D. Sculley, Sebastian Nowozin, Joshua Dillon, Balaji Lakshminarayanan, Jasper Snoek
- **Year**: 2019
- **Conference**: *Advances in Neural Information Processing Systems 32 (NeurIPS 2019)*, 13991–14002
- **Stable URL**: [https://proceedings.neurips.cc/paper/2019/hash/8558cb408c1d766213754886996287e6-Abstract.html](https://proceedings.neurips.cc/paper/2019/hash/8558cb408c1d766213754886996287e6-Abstract.html)
- **arXiv**: [arXiv:1906.02530](https://arxiv.org/abs/1906.02530)
- **Verification Status**: Full Text Verified (OA)
- **Domain Relevance**: Empirically proves that calibration worsens dramatically under dataset shift and that ensemble methods provide superior uncertainty estimation compared to single-point estimators.

### [REF-19] Mahalanobis Distance for Tabular Out-of-Distribution Detection
- **Title**: A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks
- **Authors**: Kimin Lee, Kibok Lee, Honglak Lee, Jinwoo Shin
- **Year**: 2018
- **Conference**: *Advances in Neural Information Processing Systems 31 (NeurIPS 2018)*, 7167–7177
- **Stable URL**: [https://proceedings.neurips.cc/paper/2018/hash/abdeb6f575ac5c6676b747bca8d09cc2-Abstract.html](https://proceedings.neurips.cc/paper/2018/hash/abdeb6f575ac5c6676b747bca8d09cc2-Abstract.html)
- **arXiv**: [arXiv:1807.03888](https://arxiv.org/abs/1807.03888)
- **Verification Status**: Full Text Verified (OA)
- **Domain Relevance**: Direct mathematical basis for HeartGuard AI's tabular OOD detection module, using feature covariance modeling and Mahalanobis confidence scores to flag out-of-bounds patient feature vectors.
