# HeartGuard AI: Development Progress & Project Ledger

**Current Status**: Week 2 Complete  
**Current Milestone**: Milestone 2 — Dataset Ingestion, Provenance Auditing, EDA, Preprocessing  
**Developer Commitment**: 6–8 hours/week (1 flexible working day per week)  
**Last Updated**: November 2026 (Week 2 Execution Complete)

---

## 1. Project Phase Overview

```
Milestone 1: Week 1 [COMPLETED]  - Research Foundation, Literature Review, Architecture, Documentation
Milestone 2: Week 2 [COMPLETED]  - Dataset Ingestion, Provenance Auditing, EDA, Preprocessing
Milestone 3: Week 3 [PLANNED]    - Traditional ML Baselines & Cross-Validation Benchmarks (5 Models)
Milestone 4: Week 4 [PLANNED]    - Advanced Ensembles, Optuna Tuning, SMOTENC vs. Class Weighting (5 Models)
Milestone 5: Week 5 [PLANNED]    - Cross-Dataset Generalization & External Validation Experiments
Milestone 6: Week 6 [PLANNED]    - SHAP Explainability, Probability Calibration, Subgroups, OOD Filter
Milestone 7: Week 7 [PLANNED]    - FastAPI Backend, PostgreSQL Async ORM, JWT Authentication, API Tests
Milestone 8: Week 8 [PLANNED]    - React + Vite + Tailwind Frontend, Plotly Visualizations, Clinical UI
Milestone 9: Week 9 [PLANNED]    - Docker Compose Orchestration, End-to-End Integration, Security Audit
Milestone 10: Week 10 [PLANNED]  - Final B.Tech Major Thesis, Conference Paper Draft, PPT, Viva Defense
```

---

## 2. Week 1 Task Execution Status

| Task ID | Description | Status | Verification Artifact |
| :--- | :--- | :--- | :--- |
| **TASK 1** | Inspect workspace and establish clean project root | **COMPLETED** | Repository inspected; clean root confirmed |
| **TASK 2** | Research and verify peer-reviewed academic papers | **COMPLETED** | 21 papers independently verified via PubMed / DOIs |
| **TASK 3** | Create Literature Review and Comparative Matrix | **COMPLETED** | `LITERATURE_REVIEW.md`, `LITERATURE_REVIEW_MATRIX.md` |
| **TASK 4** | Establish evidence-backed research gap across 5 areas | **COMPLETED** | `RESEARCH_GAP.md` |
| **TASK 5** | Finalize formal research problem statement | **COMPLETED** | `RESEARCH_OBJECTIVES.md` |
| **TASK 6** | Define research objectives (RO1–RO6) & questions (RQ1–RQ6)| **COMPLETED** | `RESEARCH_OBJECTIVES.md` |
| **TASK 7** | Research & document datasets, compatibility, & provenance | **COMPLETED** | `DATASET_DOCUMENTATION.md` |
| **TASK 8** | Finalize ML research methodology & anti-leakage rules | **COMPLETED** | `RESEARCH_METHODOLOGY.md` |
| **TASK 9** | Design application architecture & modular monolith | **COMPLETED** | `SYSTEM_ARCHITECTURE.md` |
| **TASK 10**| Design relational database schema & ER diagram | **COMPLETED** | `DATABASE_DESIGN.md` |
| **TASK 11**| Create project requirements & quality testing strategy | **COMPLETED** | `PROJECT_REQUIREMENTS.md`, `TESTING_STRATEGY.md` |
| **TASK 12**| Create master roadmap covering all 10 weeks | **COMPLETED** | `MASTER_ROADMAP.md` |
| **TASK 13**| Initialize Git repository and create clean project structure | **COMPLETED** | Git initialized, `.gitignore`, `README.md` created |
| **TASK 14**| Create and maintain development progress ledger | **COMPLETED** | `DEVELOPMENT_PROGRESS.md` |

---

## 3. Decisions Made (Architectural & Scientific)

1. **Dual-Pipeline Architectural Isolation**:
   - *Decision*: Strictly isolate Experiment Pipeline A (Diagnostic CAD classification on UCI multicenter collection) from Experiment Pipeline B (Prognostic 10-year incident CHD risk on Framingham educational cohort).
   - *Rationale*: Merging or cross-mapping these datasets creates an invalid hybrid task conflating cross-sectional angiographic stenosis with longitudinal prospective event incidence.
2. **Cleveland-Only Model Derivation & Frozen Zero-Shot External Transfer**:
   - *Decision*: All model development, feature selection, hyperparameter tuning, calibrator fitting, and threshold selection will be executed exclusively on Cleveland. External datasets (Hungary, Switzerland, Long Beach VA) will be strictly frozen test sets evaluated zero-shot.
   - *Rationale*: Prevents information leakage and artificial inflation of external validation performance.
3. **Common Reduced Diagnostic Set (10-Feature vs. 9-Feature Variants)**:
   - *Decision*: Adopt the standardized designation "Common Reduced Diagnostic Set", explicitly separating the 10-feature variant (with cholesterol, evaluated on Hungary and Long Beach VA with Cleveland-median imputation) from the 9-feature variant (without cholesterol, evaluated across all sites including Switzerland where cholesterol is 100% missing).
   - *Rationale*: Avoids manufacturing synthetic laboratory values on entire unmeasured cohorts while maintaining rigorous cross-center comparability.
4. **Separate Model Training per Feature Configuration**:
   - *Decision*: Full Diagnostic (13 features), Common Reduced (10 features), and Common Reduced (9 features) are trained, tuned, and calibrated as three completely separate, independent pipelines on Cleveland before being frozen.
   - *Rationale*: Models must learn optimal decision boundaries on their native feature subspace rather than having features dropped post-hoc at test time.
5. **Unverified Status for Framingham Educational CSV and Statlog Duplication**:
   - *Decision*: Mark Framingham 4,240-row educational CSV provenance and usage rights as unverified until confirmed. Keep Statlog duplicate status unverified pending Week 2 programmatic row-hash matching.
   - *Rationale*: Maintains strict scientific honesty and avoids unsupported assertions.
6. **Single Coherent Internal Validation Strategy**:
   - *Decision*: Use Repeated Stratified 5-Fold Cross-Validation (10 repeats, 50 splits) on the entire Cleveland cohort ($N=303$) with non-parametric bootstrap confidence intervals. Remove the confusing 20% internal holdout.
   - *Rationale*: Maximizes training sample efficiency and avoids underpowered validation splits.
7. **Cross-Fitted Probability Calibration with Modern API**:
   - *Decision*: Implement probability calibration using `CalibratedClassifierCV(estimator=model, cv=5)`. Evaluate calibration strictly on out-of-fold validation predictions to prevent optimistic bias.
8. **Modality-Specific Preprocessing & SMOTE Constraint**:
   - *Decision*: Numeric continuous features are median-imputed and scaled; categorical features are mode-imputed and one-hot encoded; binary features are preserved. Ordinary SMOTE after one-hot encoding is strictly prohibited; cost-sensitive class weighting is prioritized, with SMOTENC restricted to exploratory training-fold runs.
9. **Explainability Grounding (TreeSHAP Margin vs. Calibrated Probabilities)**:
   - *Decision*: TreeSHAP explains the base model's raw margin / log-odds space (where additivity holds exactly); acknowledge that non-linear sigmoid calibration does not preserve exact feature attribution rankings or signs in probability space.
10. **OOD Threshold Selection Boundary (Cleveland Only)**:
    - *Decision*: OOD distance cutoffs must be determined strictly from Cleveland training/validation data, and never tuned or adjusted using external hospital cohorts.
11. **ECG Scope Boundary**:
    - *Decision*: ECG is strictly limited to discrete tabular clinical features (`restecg`, `oldpeak`, `slope`). Waveform signal processing or file uploads are out of scope and deferred until after Week 2.

---

## 4. Academic Research Verification Status

21 peer-reviewed papers have been verified with complete bibliographic details, DOIs, and access statuses:
- **Clinical Prediction Standards**: Collins et al. (2015, *BMJ* [REF-01]), Collins et al. (2024, *BMJ* [REF-01b] TRIPOD+AI), Steyerberg & Vergouwe (2014, *Eur Heart J* [REF-02]).
- **Benchmark Datasets**: Detrano et al. (1989, *Am J Cardiol* [REF-03]), Wilson et al. (1998, *Circulation* [REF-04]).
- **Generalization & Validation**: Siontis et al. (2015, *J Clin Epidemiol* [REF-05]), Christodoulou et al. (2019, *J Clin Epidemiol* [REF-06]), Weng et al. (2017, *PLOS ONE* [REF-07]), Alaa et al. (2019, *PLOS ONE* [REF-08]).
- **Probability Calibration**: Van Calster et al. (2016, *J Clin Epidemiol* [REF-09]), Van Calster et al. (2019, *BMC Med* [REF-10]).
- **Explainable AI (SHAP)**: Lundberg & Lee (2017, *NeurIPS* [REF-11]), Lundberg et al. (2018, *Nature Biomed Eng* [REF-12]), Rudin (2019, *Nature Mach Intell* [REF-13]), Ward et al. (2020, *npj Digit Med* [REF-14]).
- **Fairness & Demographics**: Chen et al. (2021, *Annu Rev Biomed Data Sci* [REF-15]), Rajkomar et al. (2018, *Annals Intern Med* [REF-16]), Kleinberg, Mullainathan, & Raghavan (2016/2017 [REF-16b]), Woodward (2019, *Int J Env Res Pub Health* [REF-17]).
- **Uncertainty & OOD Detection**: Ovadia et al. (2019, *NeurIPS* [REF-18]), Lee et al. (2018, *NeurIPS* [REF-19]).

*Verification Summary: 100% verified. Zero unverified or fabricated citations present.*

---

## 5. Dataset Status

- **UCI Heart Disease Collection (Cleveland, Hungarian, Long Beach, Switzerland)**:
  - Official UCI repository verified (Dataset ID: 45).
  - Four sub-cohort files documented; distinct missingness patterns, class distributions, and sample sizes documented separately.
  - Frozen Cleveland-only model derivation strategy established.
- **Statlog Heart Disease (Dataset ID: 145)**:
  - Official UCI description documented (270 records, 13 numerical attributes).
  - Exact duplicate status unverified pending Week 2 programmatic row-hash matching.
- **Framingham Heart Study Educational Subset**:
  - Educational 4,240-row teaching CSV distinguished from full NHLBI study.
  - Exact institutional provenance and formal licensing marked as unverified until confirmed.
  - 10-year incident hard CHD target verified; isolated into Pipeline B.

---

## 6. Current Issues, Scientific Audit Corrections & Outstanding Questions

### Audit Corrections Applied Across Project Documentation:
1. **10-Feature vs. 9-Feature Reconciliation**: Clearly distinguished the 10-feature variant (with cholesterol) and 9-feature variant (without cholesterol) of the "Common Reduced Diagnostic Set" in `DATASET_DOCUMENTATION.md` and `RESEARCH_METHODOLOGY.md`.
2. **Removed Unsupported Claims on Missingness**: Stripped speculative historical explanations for missingness (e.g., admission protocols, surgical triage narratives, data extraction omissions). Recorded only official facts from Detrano et al. (1989) and noted that exact institutional reasons are unrecorded.
3. **Framingham Provenance & Licensing**: Explicitly marked the extraction pipeline, cohort slice, and usage rights of the 4,240-row educational CSV as unverified pending confirmation.
4. **Statlog Duplicate Status**: Kept duplicate concordance unverified pending Week 2 programmatic row-hash verification.
5. **Terminology Uniformity**: Replaced "Core Bedside Set" with "Common Reduced Diagnostic Set" across all documents.
6. **Corrected Clinical Feature Semantics**: Accurately defined `fbs` as a binary thresholded indicator ($>120\text{ mg/dL}$), and verified clinical semantics for `cp`, `restecg`, `oldpeak`, `slope`, `ca`, and `thal`.
7. **Single Coherent Internal Validation Strategy**: Adopted Repeated Stratified 5-Fold CV (10 repeats, 50 splits) on Cleveland, removing the confusing 20% internal holdout.
8. **Scikit-Learn API & Cross-Fitted Calibration**: Formulated `CalibratedClassifierCV(estimator=model, cv=5)` using the modern API; documented the exact cross-fitting procedure and enforced that calibration is evaluated strictly on predictions from observations not used to fit the corresponding base estimator or calibrator.
9. **Modality-Specific Preprocessing & SMOTE Guardrail**: Specified median imputation + scaling for numeric, mode imputation + one-hot encoding for categorical, and mode imputation + passthrough for binary. Prohibited ordinary SMOTE after one-hot encoding; preferred cost-sensitive class weighting and restricted SMOTENC to exploratory training folds.
10. **Clean Model Hyperparameter Grids**: Specified consistent, regularized search spaces across all 10 algorithms.
11. **Frozen Decision Thresholds**: Pre-specified Youden's $J$ threshold optimization on Cleveland out-of-fold predictions and froze $\tau^*$ before zero-shot external evaluation.
12. **Rigorous Kleinberg et al. Impossibility Theorem**: Accurately described the specific theorem in Kleinberg, Mullainathan, & Raghavan (2016/2017 [REF-16b]), proving that calibration within groups, balance for the positive class, and balance for the negative class cannot simultaneously hold under unequal prevalence unless prediction is perfect; avoided overgeneralizing to every arbitrary fairness metric combination.
13. **TreeSHAP Raw Margin vs. Calibrated Output Correction**: Corrected the mathematical claim: TreeSHAP computes exact attributions in base model log-odds space, but these attributions do not automatically preserve signs or rank-orderings in probability space due to the non-linear sigmoid transformation. Clearly distinguished base-model explanations from post-calibration probabilities.
14. **Statistical Uncertainty Methodology**: Documented that 50 repeated CV folds are statistically correlated and cannot be treated as independent observations; specified patient-level non-parametric bootstrap CIs ($B=1,000$); barred DeLong testing on pooled repeated-CV predictions; and documented the wide variance and limitations of CIs on small demographic subgroups.
15. **OOD Threshold Selection Boundary (Cleveland Only)**: Mandated that OOD distance cutoffs must be determined strictly from Cleveland training/validation data, and never tuned or adjusted using external hospital cohorts.
16. **Separate Training of Feature Configurations**: Mandated that Full Diagnostic (13 features), Common Reduced (10 features), and Common Reduced (9 features) are trained, tuned, and calibrated as three completely separate, independent pipelines on Cleveland before being frozen.

### Outstanding Scientific Questions for Week 2:
- **Q1 (Statlog Duplication Concordance)**: Exactly how many of Statlog's 270 rows match Cleveland's 303 rows under feature tolerance matching, and how are missing values in Cleveland represented in Statlog?
- **Q2 (Non-Random Missingness Handling in Switzerland)**: In the Swiss dataset, cholesterol is recorded as 0 across all 123 rows. The underlying cause is not documented in the archival literature and remains unverified. When evaluating the frozen Cleveland models zero-shot, how does the 10-feature model with Cleveland-median cholesterol imputation compare empirically to the 9-feature model that omits cholesterol?
- **Q3 (OOD False-Alarm Rate on Cleveland Folds)**: What is the empirical false-positive rejection rate of the 99th percentile Cleveland-derived Ledoit-Wolf Mahalanobis distance cutoff when evaluated on out-of-fold Cleveland patients, and what proportion of external hospital patients (Hungary, Long Beach, Switzerland) are flagged as out-of-distribution?
- **Q4 (Framingham Educational CSV License Confirmation)**: What are the formal data transfer terms and licensing constraints governing the 4,240-row educational teaching dataset?

---

## 7. Week 1 Acceptance Criteria Checklist

- [x] At least 10–15 relevant papers researched and verified (21 verified).
- [x] Literature review matrix created (`LITERATURE_REVIEW_MATRIX.md`).
- [x] Research gap is evidence-backed and cautiously worded (`RESEARCH_GAP.md`).
- [x] Research objectives (RO1–RO6) and questions (RQ1–RQ6) documented (`RESEARCH_OBJECTIVES.md`).
- [x] The two prediction experiments are clearly separated into isolated pipelines.
- [x] Dataset provenance, compatibility issues, and feature configurations documented (`DATASET_DOCUMENTATION.md`).
- [x] Research methodology established with anti-leakage rules and single coherent validation scheme (`RESEARCH_METHODOLOGY.md`).
- [x] Architecture documented with modular monolith blueprints (`SYSTEM_ARCHITECTURE.md`).
- [x] Database design documented with PostgreSQL schema and ER diagram (`DATABASE_DESIGN.md`).
- [x] Requirements (`PROJECT_REQUIREMENTS.md`) and testing strategy (`TESTING_STRATEGY.md`) exist.
- [x] All 14 required project files created in project root.
- [x] The 10-week master roadmap exists (`MASTER_ROADMAP.md`).
- [x] `DEVELOPMENT_PROGRESS.md` maintained and updated with audit corrections.
- [x] No fabricated research, results, or unsupported claims are present.

---

## 8. Week 2 Execution Results

### Week 2 Task Status

| Task ID | Description | Status | Artifact |
|:--------|:-----------|:-------|:---------|
| **W2-T1** | Create project directories (`data/raw/`, `data/processed/`, `notebooks/`, `src/pipelines/`, `scripts/`) | **COMPLETED** | Directory structure created |
| **W2-T2** | Download Cleveland raw dataset | **COMPLETED** | `data/raw/uci/cleveland.data` (SHA256: a74b7ef...) |
| **W2-T3** | Download Hungary raw dataset | **COMPLETED** | `data/raw/uci/hungary.data` (SHA256: d1ad108...) |
| **W2-T4** | Download Switzerland raw dataset | **COMPLETED** | `data/raw/uci/switzerland.data` (SHA256: 834a405...) |
| **W2-T5** | Download VA Long Beach raw dataset | **COMPLETED** | `data/raw/uci/va.data` (SHA256: e7c93d8...) |
| **W2-T6** | Download Statlog heart dataset | **COMPLETED** | `data/raw/statlog/heart.dat` (SHA256: f5f3b42...) |
| **W2-T7** | Retrieve Framingham educational CSV | **COMPLETED** | `data/raw/framingham/framingham.csv` (SHA256: 2c0e57d...) — provenance UNRESOLVED |
| **W2-T8** | Record SHA-256 checksums and provenance | **COMPLETED** | `data/raw/CHECKSUMS.md` |
| **W2-T9** | Statlog vs. Cleveland programmatic duplicate verification | **COMPLETED + AUDITED** | `reports/STATLOG_DUPLICATE_REPORT.md` (corrected); `reports/STATLOG_AUDIT_DETAIL.csv` |
| **W2-T10** | Framingham provenance investigation | **COMPLETED** | `reports/DATASET_PROVENANCE_REPORT.md` (marked UNRESOLVED) |
| **W2-T11** | Automated EDA (dimensions, types, missingness, distributions, correlations, site comparisons, class imbalance) | **COMPLETED** | `reports/EXPLORATORY_DATA_ANALYSIS_REPORT.md` |
| **W2-T12** | Missingness report | **COMPLETED** | `reports/MISSINGNESS_REPORT.md` |
| **W2-T13** | Dataset compatibility report | **COMPLETED** | `reports/DATASET_COMPATIBILITY_REPORT.md` |
| **W2-T14** | Build reusable preprocessing pipelines (13F / 10F / 9F / Framingham) | **COMPLETED** | `src/pipelines/preprocessors.py` |
| **W2-T15** | Preprocessing documentation | **COMPLETED** | `reports/PREPROCESSING_DOCUMENTATION.md` |
| **W2-T16** | Anti-data-leakage tests (7 tests) | **COMPLETED** | `tests/test_leakage.py` — 7/7 PASS |
| **W2-T18** | Statlog audit: deep verification with NaN/complete-row classification | **COMPLETED** | `scripts/audit_statlog_duplicates.py`; 22 automated tests 22/22 PASS |
| **W2-T17** | EDA notebook | **COMPLETED** | `notebooks/01_exploratory_data_analysis.ipynb` |

**Week 2 Completion: 100%**

---

### Key Week 2 Findings

1. **Statlog Duplicate Status (Q1 RESOLVED — CORRECTED by Week 2 Audit):**

   Initial report (before audit): claimed NaN values explain all 33 unmatched Cleveland rows. **This was incorrect.**

   Corrected findings from `scripts/audit_statlog_duplicates.py` (22 automated tests, all pass):

   | Metric | Verified Count |
   |--------|---------------|
   | Cleveland rows with exact Statlog match | **270 of 303** |
   | Statlog rows with a Cleveland counterpart | **270 of 270 (100%)** |
   | Unmatched Cleveland rows -- contain NaN predictor(s) | **6** |
   | Unmatched Cleveland rows -- complete (no NaN), no Statlog counterpart | **27** |
   | Target label inconsistencies among 270 matched rows | **0** |

   **The 27 complete unmatched rows are Cleveland-only records with no Statlog equivalent.** They are not explained by NaN. The 27 rows are indices 274-301 (excluding 287) — the last ~28 rows of the Cleveland file which appear to be an augmented subset not present in Statlog.

   **Corrected conclusion:** Statlog is a proper subset of Cleveland (270 of 303 rows), but the relationship is NOT simply "complete cases of Cleveland." Cleveland contains 27 additional complete rows that are absent from Statlog. Feature coding is numerically identical across all 13 predictors. All 270 matched rows have consistent binary target labels under the documented mapping (Cleveland num>=1 <-> Statlog target==2).

   **Critical implication:** Statlog must NOT be used as an external validation set. All 270 of its rows exist in Cleveland — using it as a test set while training on Cleveland constitutes data leakage.

2. **Framingham Provenance (Q4 PARTIALLY RESOLVED):**
   - File retrieved from GitHub educational mirror (GauravPadawe repository).
   - 4,238 rows, 16 columns, target: `TenYearCHD`.
   - Formal redistribution licence from NHLBI BIOLINCC cannot be confirmed from GitHub mirror alone.
   - **Status: UNRESOLVED** — marked for provenance confirmation before use in published results.

3. **Switzerland cholesterol zeros:**
   - Switzerland has a high proportion of zero-coded `chol` values. Cause is NOT documented in the UCI repository. No unsupported clinical attribution is made.

4. **Anti-Leakage Tests:** All 7 pytest assertions pass, confirming no data from test/validation folds leaks into training-fold imputation or scaling statistics.

---

### Week 2 Unresolved Issues

| ID | Issue | Status |
|----|-------|--------|
| U1 | Framingham educational CSV redistribution licence | UNRESOLVED -- requires NHLBI BIOLINCC verification |
| U2 | Statlog/Cleveland: 27 complete Cleveland rows absent from Statlog -- nature of this subset unknown | OPEN -- structure of Cleveland file (rows 274-301) vs. Statlog origin not explained by available documentation |
| U3 | Switzerland cholesterol zero-coding cause | UNVERIFIED -- not documented in UCI repository |
| U4 | Q2: empirical comparison of 10F vs. 9F models on Switzerland | DEFERRED to Week 5 external validation |
| U5 | Q3: OOD false-alarm rate on Cleveland folds | DEFERRED to Week 6 |

---

## 9. Next Session Plan (Week 3)

When Week 3 begins, execute:
1. Implement 5 baseline ML models on Cleveland: Logistic Regression, Decision Tree, Random Forest, SVM, k-NN.
2. Run Repeated Stratified 5-Fold CV (10 repeats, 50 splits) using the three feature configurations separately.
3. Collect AUC-ROC, AUC-PR, accuracy, F1, sensitivity, specificity per fold per model.
4. Compute non-parametric bootstrap confidence intervals (B=1,000) for each metric.
5. Document results in `reports/BASELINE_MODEL_RESULTS.md`.
6. Ensure no hyperparameter tuning occurs in Week 3 (baselines with default settings first).
7. Commit results to Git.
