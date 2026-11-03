# HeartGuard AI: Research Objectives and Research Questions

**Project Title**: HeartGuard AI: An Explainable, Calibrated and Cross-Dataset Machine Learning Framework for Heart Disease Risk Prediction  
**Scope**: 10-Week Engineering & Applied Machine Learning Research Study  
**Standard**: TRIPOD Statement (2015) & TRIPOD+AI Statement (2024 [REF-01b])

---

## 1. Formal Problem Statement

Cardiovascular disease remains the leading cause of global mortality. While academic literature abounds with machine learning models reporting high classification accuracy ($>85\%$) on isolated benchmark subsets such as the UCI Cleveland dataset, these metrics reflect optimistic internal resubstitution or random-split testing rather than authentic clinical utility.

In real-world healthcare environments, clinical decision-support models suffer from critical deficiencies:
1. **Validation Failure**: Models degrade sharply when evaluated on independent patient populations (covariate shift and dataset shift).
2. **Endpoint Conflation**: Studies frequently conflate cross-sectional diagnostic detection of coronary stenosis with 10-year prognostic incidence of cardiovascular events.
3. **Miscalibration**: Algorithms output raw decision scores that fail to reflect empirical disease probabilities, leading to miscalculated clinical interventions.
4. **Opaque & Misinterpreted Attributions**: Post-hoc explanations are rarely provided, or when provided, are misinterpreted as causal mechanisms rather than statistical correlations.
5. **Demographic Inequity & Silent Outlier Failure**: Aggregate evaluation conceals diagnostic disparities across biological sex and age cohorts, while systems lack mechanisms to detect out-of-distribution inputs.

HeartGuard AI formulates an empirical research investigation and open-source software framework to systematically evaluate these interconnected challenges across two distinct clinical pipelines.

---

## 2. Research Objectives (RO)

* **RO1: Reproducible Pipeline Architecture & Strict Dataset Segregation**  
  Design, document, and implement an end-to-end reproducible machine learning data processing pipeline that strictly separates cross-sectional diagnostic coronary artery disease (UCI Multicenter Collection) from longitudinal 10-year prognostic cardiovascular risk (Framingham Heart Study), eliminating data leakage and endpoint conflation.

* **RO2: Multi-Model Benchmark Under Robust Validation**  
  Implement and benchmark five traditional algorithms (Logistic Regression, Decision Trees, Random Forest, K-Nearest Neighbors, Support Vector Classifier) and five advanced ensemble/boosting algorithms (Gradient Boosting, XGBoost, LightGBM, CatBoost, Stacking Classifier) using Repeated Stratified 5-Fold Cross-Validation on the Cleveland derivation dataset, with non-parametric bootstrap confidence intervals to account for small sample size uncertainty ($N=303$).

* **RO3: Systematic Cross-Dataset Generalization Evaluation Under Frozen Parameters**  
  Quantify the generalization gap by developing, tuning, and calibrating models exclusively on the primary Cleveland dataset, freezing all preprocessing transformations, model weights, calibrator parameters, and decision thresholds, and evaluating zero-shot external performance on independent cohorts (Hungarian Institute of Cardiology, VA Long Beach, Switzerland) across Full (13-feature), Common Reduced (10-feature with cholesterol), and Common Reduced (9-feature without cholesterol) configurations without any external tuning or re-calibration.

* **RO4: Local and Global Explainability via SHAP**  
  Integrate SHapley Additive exPlanations (KernelSHAP and TreeSHAP) to provide both population-level feature importance hierarchies and individual patient-level risk attribution waterfall plots, accompanied by explicit non-causal interpretability guidelines.

* **RO5: Probability Calibration Assessment and Recalibration**  
  Assess model calibration using the Van Calster calibration hierarchy, computing Brier scores, calibration slopes, and calibration-in-the-large intercepts. Implement and compare post-hoc calibration methods (Platt Scaling and Isotonic Regression) evaluated on dedicated validation folds.

* **RO6: Subgroup Performance Auditing and Out-of-Distribution Guardrails**  
  Conduct disaggregated performance audits across demographic strata (Sex: Male vs. Female; Age: $<50$, $50–65$, $>65$) to detect diagnostic disparities. Design and implement a multivariate Mahalanobis distance out-of-distribution (OOD) filter and predictive entropy thresholds to flag anomalous clinical inputs.

---

## 3. Research Questions (RQ) and Hypotheses

### RQ1: Generalization Gap Across Independent Cohorts
* **Question**: *How does model performance change when evaluated on an independent geographical/institutional dataset compared to a standard stratified random holdout from the training dataset?*
* **Hypothesis**: Discrimination (ROC-AUC) will experience a statistically significant drop ($\Delta\text{AUC} \le -0.05$) when transferring from Cleveland to independent cohorts (Hungary, Long Beach, Switzerland), driven by differences in baseline demographic covariates and institutional clinical protocols.
* **Academic Objectivity Rule**: If models retain discrimination without significant loss, or if simpler linear models generalize better than complex ensembles, this finding will be reported as an authentic empirical discovery.

### RQ2: Model Architecture and Comparative Generalization
* **Question**: *Which machine learning algorithms demonstrate comparatively superior generalization resilience when transferred across compatible clinical datasets?*
* **Hypothesis**: Heavily regularized models (e.g., $L_2$-penalized Logistic Regression) and moderately constrained tree ensembles (e.g., CatBoost / Random Forest with controlled tree depth) will exhibit smaller generalization degradation than unconstrained deep trees or high-capacity boosting algorithms that overfit idiosyncratic training-cohort patterns.
* **Academic Objectivity Rule**: Ensembles will not be presumed superior. Actual ranking and relative degradation across all 10 algorithms will be reported transparently.

### RQ3: Calibration Impact on Probability Reliability
* **Question**: *Does post-hoc probability calibration (Platt scaling / Isotonic regression) improve the clinical reliability of predicted probabilities without compromising discriminative ability?*
* **Hypothesis**: Post-hoc calibration will significantly reduce Brier score and adjust the calibration slope toward 1.0 on internal validation; however, calibration parameters learned on the training cohort will exhibit calibration drift when transferred across datasets.
* **Academic Objectivity Rule**: If isotonic regression overfits on smaller folds or Platt scaling fails under covariate shift, this limitation will be documented.

### RQ4: Explainability Utility and Clinical Decision Support
* **Question**: *How do local and global SHAP attributions characterize model decision boundaries across traditional and ensemble algorithms, and do feature importance rankings align with established clinical knowledge?*
* **Hypothesis**: SHAP values will reveal that physiological markers (chest pain type, ST-segment depression, maximum heart rate achieved, fluoroscopy vessels) consistently dominate predictive decisions across top models, providing transparent justification for individual predictions while maintaining fidelity to model logic.
* **Academic Objectivity Rule**: Feature attributions will be critically appraised; any counter-intuitive negative attributions (e.g., paradoxes caused by collinearity) will be explicitly analyzed rather than concealed.

### RQ5: Demographic Parity and Subgroup Disparities
* **Question**: *Does model performance (sensitivity, specificity, calibration) vary systematically across demographic subgroups defined by biological sex and age brackets?*
* **Hypothesis**: Due to historical class prevalence and symptom presentation differences, models trained on predominantly male cohorts will exhibit lower sensitivity and higher false-negative rates in female patients.
* **Academic Objectivity Rule**: Both positive parity and persistent disparities will be documented objectively according to Equalized Odds criteria.

### RQ6: Detection of Unsupported and Out-of-Distribution Inputs
* **Question**: *How effectively can a multivariate Mahalanobis distance metric and prediction entropy thresholds detect anomalous, physiologically improbable, or out-of-distribution clinical vectors?*
* **Hypothesis**: Class-conditional Mahalanobis distance computed on standardized continuous features will identify extreme and anomalous patient vectors, providing a reliable guardrail to alert clinicians when inference cannot be safely trusted.
* **Academic Objectivity Rule**: False rejection rates on borderline valid patients will be evaluated to verify that the guardrail does not over-reject unusual but legitimate clinical cases.

---

## 4. Methodological Alignment Matrix

| Research Objective | Primary Research Question | Validation Methodology | Key Output Artifact |
| :--- | :--- | :--- | :--- |
| **RO1** (Pipeline & Segregation) | Cross-Pipeline Integrity | Code modularity, schema validation, zero-leakage pipeline | `src/data/`, `DATASET_DOCUMENTATION.md` |
| **RO2** (10-Model Benchmark) | RQ2 (Comparative Model Performance) | Repeated Stratified 5-Fold CV (10 repeats), Hyperparameter search | Benchmark leaderboard, Confusion matrices |
| **RO3** (Cross-Dataset Evaluation) | RQ1 (Generalization Gap) | Train on Cleveland $\rightarrow$ Test on Hungary, Long Beach, Switzerland | Generalization gap tables, $\Delta\text{AUC}$ plots |
| **RO4** (SHAP Explainability) | RQ4 (Attribution & Fidelity) | Global summary beeswarm, dependence plots, individual waterfall | `src/explainability/`, interactive SHAP dashboard |
| **RO5** (Probability Calibration) | RQ3 (Calibration Reliability) | Calibration curves, Brier score, Platt / Isotonic methods | Calibration comparison curves, Brier matrix |
| **RO6** (Subgroup & OOD Guardrails)| RQ5 (Subgroup Equity) & RQ6 (OOD Safety) | Stratified sex/age metrics, Mahalanobis distance filter | Fairness audit reports, OOD filter module |

