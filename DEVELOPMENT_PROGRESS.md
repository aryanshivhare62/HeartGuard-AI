# HeartGuard AI: Development Progress & Project Ledger

**Current Status**: Week 1 Complete (Post-Scientific Audit & Reconciliation)  
**Current Milestone**: Milestone 1 — Research Foundation, Literature Verification, System Design & Architecture  
**Developer Commitment**: 6–8 hours/week (1 flexible working day per week)  
**Last Updated**: November 2026 (Comprehensive Scientific Audit Applied)

---

## 1. Project Phase Overview

```
Milestone 1: Week 1 [COMPLETED]  - Research Foundation, Literature Review, Architecture, Documentation
Milestone 2: Week 2 [PLANNED]    - Dataset Ingestion, Provenance Auditing, EDA, Preprocessing
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
4. **Unverified Status for Framingham Educational CSV and Statlog Duplication**:
   - *Decision*: Mark Framingham 4,240-row educational CSV provenance and usage rights as unverified until confirmed. Keep Statlog duplicate status unverified pending Week 2 programmatic row-hash matching.
   - *Rationale*: Maintains strict scientific honesty and avoids unsupported assertions.
5. **Single Coherent Internal Validation Strategy**:
   - *Decision*: Use Repeated Stratified 5-Fold Cross-Validation (10 repeats, 50 splits) on the entire Cleveland cohort ($N=303$) with non-parametric bootstrap confidence intervals. Remove the confusing 20% internal holdout.
   - *Rationale*: Maximizes training sample efficiency and avoids underpowered validation splits.
6. **Cross-Fitted Probability Calibration**:
   - *Decision*: Implement probability calibration via out-of-fold cross-fitting (`CalibratedClassifierCV(cv=5)`) to eliminate optimistic calibration bias.
7. **Modality-Specific Preprocessing & SMOTE Constraint**:
   - *Decision*: Numeric continuous features are median-imputed and scaled; categorical features are mode-imputed and one-hot encoded; binary features are preserved. Ordinary SMOTE after one-hot encoding is strictly prohibited; cost-sensitive class weighting is prioritized, with SMOTENC restricted to exploratory training-fold runs.
8. **Explainability Grounding (TreeSHAP Margin vs. Calibrated Probabilities)**:
   - *Decision*: TreeSHAP explains the base model's raw margin / log-odds space (where additivity holds exactly); explain how strictly monotonic Platt scaling preserves feature direction and rank ordering.
9. **ECG Scope Boundary**:
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
2. **Removed Unsupported Claims**: Stripped speculative historical explanations for missingness (e.g., admission protocols, surgical triage narratives). Recorded only official facts from Detrano et al. (1989) and noted that exact institutional reasons are unrecorded.
3. **Framingham Provenance & Licensing**: Explicitly marked the extraction pipeline, cohort slice, and usage rights of the 4,240-row educational CSV as unverified pending confirmation.
4. **Statlog Duplicate Status**: Kept duplicate concordance unverified pending Week 2 programmatic row-hash verification.
5. **Terminology Uniformity**: Replaced "Core Bedside Set" with "Common Reduced Diagnostic Set" across all documents.
6. **Corrected Clinical Feature Semantics**: Accurately defined `fbs` as a binary thresholded indicator ($>120\text{ mg/dL}$), and verified clinical semantics for `cp`, `restecg`, `oldpeak`, `slope`, `ca`, and `thal`.
7. **Single Coherent Internal Validation Strategy**: Adopted Repeated Stratified 5-Fold CV (10 repeats, 50 splits) on Cleveland, removing the confusing 20% internal holdout.
8. **Cross-Fitted Calibration**: Formulated `CalibratedClassifierCV(cv=5)` to fit calibrators strictly on out-of-fold validation predictions.
9. **Modality-Specific Preprocessing & SMOTE Guardrail**: Specified median imputation + scaling for numeric, mode imputation + one-hot encoding for categorical, and mode imputation + passthrough for binary. Prohibited ordinary SMOTE after one-hot encoding; preferred cost-sensitive class weighting and restricted SMOTENC to exploratory training folds.
10. **Clean Model Hyperparameter Grids**: Specified consistent, regularized search spaces across all 10 algorithms.
11. **Frozen Decision Thresholds**: Pre-specified Youden's $J$ threshold optimization on Cleveland out-of-fold predictions and froze $\tau^*$ before zero-shot external evaluation.
12. **Fairness Impossibility Theorem**: Cited Kleinberg et al. (2016/2017 [REF-16b]) and Rajkomar et al. (2018 [REF-16]) proving the mathematical impossibility of simultaneous Equalized Odds and Predictive Parity under unequal prevalence.
13. **Clarified TreeSHAP vs. Calibrated Output**: Documented that TreeSHAP computes exact attributions in base log-odds space, and explained how monotonic Platt scaling preserves attribution directions and rank ordering.
14. **Separation of Primary vs. Exploratory Analyses**: Formally partitioned pre-specified confirmatory benchmarks from exploratory analyses (SMOTENC, threshold sensitivity, OOD cutoffs).

### Outstanding Scientific Questions for Week 2:
- **Q1 (Statlog Duplication Concordance)**: Exactly how many of Statlog's 270 rows match Cleveland's 303 rows under feature tolerance matching, and how are missing values in Cleveland represented in Statlog?
- **Q2 (Non-Random Missingness Handling in Switzerland)**: In the Swiss dataset, cholesterol is recorded as 0 across all 123 rows (clinical omission, not true zero cholesterol). When evaluating the frozen Cleveland model zero-shot, how sensitive is model discrimination to imputing Cleveland's training median versus evaluating the 9-feature model that omits cholesterol?
- **Q3 (OOD Threshold Tuning via Natural Dataset Shift)**: Does a 99th percentile $\chi^2$ cutoff on continuous features produce acceptable false-alarm rates on held-out Cleveland validation folds, or does it require empirical thresholding based on the natural distribution shift observed in the Hungarian cohort?
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

## 8. Next Session Plan (Revised Week 2 Plan)

When Week 2 begins, the agent and developer will execute:
1. Ingest raw datasets into `data/raw/` (UCI Cleveland, Hungarian, Long Beach, Switzerland; Statlog; Framingham educational CSV).
2. Execute programmatic row-hash matching script between Statlog and Cleveland to resolve Question Q1.
3. Conduct automated Exploratory Data Analysis (EDA) scripts computing distributions, skewness, missingness maps, and correlation heatmaps per site.
4. Formulate the three feature configurations (13-feature Full Diagnostic Set, 10-feature Common Reduced Set with cholesterol, 9-feature Common Reduced Set without cholesterol) with statistical justification.
5. Author scikit-learn custom preprocessing transformers (`src/pipelines/preprocessors.py`) strictly isolating fold-level imputation and scaling.
6. Verify anti-leakage test assertions on mock splits.
