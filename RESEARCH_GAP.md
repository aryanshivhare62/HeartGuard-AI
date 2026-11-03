# HeartGuard AI: Evidence-Backed Research Gap Analysis

**Document Status**: Verified against Peer-Reviewed Empirical Literature  
**Date**: November 2026 (Week 1 Research Phase)  
**Lead Investigator**: Principal ML Research Engineer & Data Scientist

---

## 1. Primary Research Gap Statement

> **Working Hypothesis**:  
> *"Existing heart disease prediction implementations frequently emphasize classification accuracy on limited benchmark datasets. There is a need for a reproducible and systematically evaluated framework that investigates cross-dataset generalization, probability calibration, explainability, and subgroup performance together, while explicitly accounting for differences in dataset populations and outcome definitions."*

### Empirical Verification Status: **CONFIRMED & EVIDENCE-BACKED**
The extensive literature review conducted in Week 1 validates every element of this hypothesis. The research gap is not an informal intuition; it is directly supported by empirical findings across meta-analyses, epidemiological investigations, and computer science literature.

---

## 2. In-Depth Investigation Across the Five Core Research Dimensions

### Dimension A: Cross-Dataset Generalization and External Validation
* **The Problem in Existing Practice**:  
  The overwhelming majority of published heart disease prediction papers (particularly those utilizing the UCI Cleveland or Kaggle heart disease repositories) report performance derived exclusively from a single random train/test split or $k$-fold cross-validation on the same dataset (TRIPOD Type 1b). Such studies frequently claim test accuracy exceeding 88–95%.
* **Empirical Evidence from Literature**:  
  - **Siontis et al. (2015, *J Clin Epidemiol*) [REF-05]**: In a systematic meta-epidemiological evaluation of 127 prediction models, only 17% were externally validated by independent investigators. When external validation was performed, the median drop in discriminative ability ($\Delta\text{AUC}$) was **$-0.05$** ($p < 0.001$). Not a single study demonstrated a discrimination increase of $\ge 0.03$.
  - **Detrano et al. (1989, *Am J Cardiol*) [REF-03]**: In the original multicenter derivation of the UCI heart disease dataset, a diagnostic algorithm developed on the Cleveland Clinic cohort experienced significant performance variability when transferred across European centers (Budapest and Zurich), reflecting differences in referral bias and underlying disease prevalence.
* **The Root Causes of Cross-Dataset Failure**:  
  1. *Covariate Shift*: Differences in baseline demographic distributions (e.g., age distributions, sex ratios) across hospitals.
  2. *Concept Drift & Spectrum Bias*: Cleveland enrolled patients referred for diagnostic coronary arteriography due to suspected angina; other centers enrolled post-infarction or rehabilitation patients.
  3. *Outcome Conflation*: Many online pipelines merge the Framingham dataset (predicting 10-year incident coronary events in asymptomatic community individuals) with UCI datasets (predicting cross-sectional anatomical stenosis $\ge 50\%$). This is an invalid conflation of diagnostic versus prognostic targets.

---

### Dimension B: Explainability and Its Clinical Limitations
* **The Problem in Existing Practice**:  
  Machine learning papers increasingly append feature importance plots (e.g., default tree feature importances or raw SHAP summary plots) as an afterthought. These are frequently accompanied by unjustified assertions that the top-ranked features represent the most promising "therapeutic targets" or "causes" of heart disease.
* **Empirical Evidence from Literature**:  
  - **Lundberg & Lee (2017, *NeurIPS*) [REF-11]** and **Lundberg et al. (2018, *Nature Biomed Eng*) [REF-12]**: SHAP provides a mathematically sound game-theoretic attribution for local predictions ($\sum \phi_i(x) = f(x) - \mathbb{E}[f(X)]$), ensuring local accuracy and consistency.
  - **Rudin (2019, *Nature Machine Intell*) [REF-13]**: Post-hoc explanations of black-box models do not explain how the underlying biological reality works; they merely explain how an unconstrained black-box mathematical surrogate approximated its training data. Furthermore, correlated features can lead to misleading attribution splits.
* **The Non-Causality Principle**:  
  SHAP values quantify statistical *association* conditional on the model architecture and feature space, **not biological causation**. For example, high ST-segment depression (`oldpeak`) or exercise-induced angina (`exang`) can be strong predictive indicators of coronary stenosis, but suppressing ST-depression artificially (e.g., through digital ECG filtering) will not cure the underlying coronary occlusion. HeartGuard AI explicitly integrates non-causal interpretability warnings.

---

### Dimension C: Probability Calibration in Clinical Risk Prediction
* **The Problem in Existing Practice**:  
  Nearly all machine learning papers benchmark models using threshold-dependent classification metrics (Accuracy, F1-score) or rank-order metrics (ROC-AUC). Virtually none assess or report **probability calibration** (TRIPOD Item 16). Consequently, an ensemble model might achieve an ROC-AUC of 0.88 while predicting probabilities that are either clustered near extremes (0.01 / 0.99) or systematically biased.
* **Empirical Evidence from Literature**:  
  - **Van Calster et al. (2016, 2019, *J Clin Epidemiol*, *BMC Medicine*) [REF-09, REF-10]**: Clinical decisions are made at specific risk thresholds (e.g., the 7.5% or 10% 10-year CVD risk threshold for prescribing statin therapy). If a model's predicted probability is 8% but the true empirical risk in patients with that score is 25%, the patient is denied life-saving preventative treatment. High discrimination is insufficient; **moderate calibration** is mandatory for safe clinical deployment.
  - **Ovadia et al. (2019, *NeurIPS*) [REF-18]**: Standard post-hoc calibration methods (Platt scaling, temperature scaling) degrade severely under dataset shift. When a model trained on Cleveland is transferred to an external site, calibration intercept and slope drift significantly.
* **HeartGuard AI Contribution**:  
  HeartGuard AI integrates calibration directly into the evaluation pipeline by reporting Brier scores, calibration slopes, calibration-in-the-large intercepts, and flexible loess/spline calibration curves alongside ROC-AUC and PR-AUC.

---

### Dimension D: Demographic Fairness and Subgroup Disparities
* **The Problem in Existing Practice**:  
  Evaluation metrics are almost uniformly computed at the aggregate dataset level. An overall test accuracy of 86% may conceal a severe disparity where male patients achieve 91% sensitivity while female patients achieve only 68% sensitivity.
* **Empirical Evidence from Literature**:  
  - **Woodward (2019, *Int J Environ Res Pub Health*) [REF-17]**: Women suffer from a systemic diagnostic disadvantage in cardiovascular medicine. Female presentations of coronary ischemia frequently involve microvascular angina, atypical symptom profiles, and different age-adjusted cholesterol trajectories compared to the classic crushing chest pain exhibited by middle-aged males.
  - **Rajkomar et al. (2018, *Annals Intern Med*) [REF-16]** and **Chen et al. (2021, *Annu Rev Biomed Data Sci*) [REF-15]**: Aggregate metrics obscure disparate error rates across protected demographic attributes. Furthermore, mathematical trade-offs prove that equalizing false positive rates and false negative rates across groups with different base prevalence is impossible unless the model is a perfect predictor.
* **HeartGuard AI Contribution**:  
  HeartGuard AI incorporates automated demographic subgroup auditing across Sex (Male vs. Female) and Age brackets ($<50$, $50–65$, $>65$), reporting stratified Sensitivity, Specificity, and Equalized Odds differentials.

---

### Dimension E: Predictive Uncertainty and Out-of-Distribution (OOD) Detection
* **The Problem in Existing Practice**:  
  Standard ML classifiers (e.g., LogisticRegression, XGBClassifier) operate on the closed-world assumption: any feature vector presented to `.predict_proba()` will receive a probability, even if the values are biologically impossible (e.g., systolic BP = 400 mmHg) or far outside the joint distribution of the training data.
* **Empirical Evidence from Literature**:  
  - **Lee et al. (2018, *NeurIPS*) [REF-19]**: Out-of-distribution inputs can be effectively detected by calculating class-conditional Mahalanobis distances in the feature representation space.
  - **Ovadia et al. (2019, *NeurIPS*) [REF-18]**: Classifiers become overconfident and generate unreliable uncertainty estimates when presented with shifted or anomalous distributions.
* **HeartGuard AI Contribution**:  
  HeartGuard AI implements a two-tier guardrail system:
  1. *Clinical Sanity Validation*: Hard physiological boundary checks rejecting impossible inputs.
  2. *Multivariate Mahalanobis Distance & Entropy Filter*: Flagging test samples that fall beyond the empirical distribution envelope of the training cohort, providing explicit warnings rather than deceptive high-confidence predictions.

---

## 3. Synthesis: Why an Integrated Framework Is Needed

The literature reveals that these five dimensions cannot be solved in isolation:
- An ensemble model that achieves high accuracy may fail completely in probability calibration.
- A well-calibrated model on a single hospital dataset may suffer calibration collapse upon external validation.
- A model with good overall cross-dataset discrimination may exhibit unacceptable false-negative rates in women.
- Post-hoc SHAP explanations may lead clinicians into unwarranted causal assumptions if not paired with transparency warnings.
- The entire system can be corrupted if anomalous patient data is evaluated without an out-of-distribution guardrail.

Therefore, the primary contribution of **HeartGuard AI** is the construction, systematic evaluation, and open-source dissemination of a **reproducible, integrated framework** that addresses all five dimensions together in a transparent, dual-pipeline clinical prediction architecture.
