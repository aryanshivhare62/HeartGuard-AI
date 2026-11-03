# HeartGuard AI: Development Progress & Project Ledger

**Current Status**: Week 1 Complete (Post-Scientific Audit)  
**Current Milestone**: Milestone 1 — Research Foundation, Literature Verification, System Design & Architecture  
**Developer Commitment**: 6–8 hours/week (1 flexible working day per week)  
**Last Updated**: November 2026 (Scientific Audit Applied)

---

## 1. Project Phase Overview

```
Milestone 1: Week 1 [COMPLETED]  - Research Foundation, Literature Review, Architecture, Documentation
Milestone 2: Week 2 [PLANNED]    - Dataset Ingestion, Provenance Auditing, EDA, Preprocessing
Milestone 3: Week 3 [PLANNED]    - Traditional ML Baselines & Cross-Validation Benchmarks (5 Models)
Milestone 4: Week 4 [PLANNED]    - Advanced Ensembles, Optuna Tuning, SMOTE vs. Class Weighting (5 Models)
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
| **TASK 2** | Research and verify peer-reviewed academic papers | **COMPLETED** | 20 papers independently verified via PubMed / DOIs |
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
2. **Cleveland-Only Model Derivation & Frozen Cross-Dataset Testing**:
   - *Decision*: All model development, feature selection, hyperparameter tuning, calibrator fitting, and threshold selection will be executed exclusively on Cleveland. External datasets (Hungary, Switzerland, Long Beach VA) will be strictly frozen test sets evaluated zero-shot.
   - *Rationale*: Prevents information leakage and artificial inflation of external validation performance.
3. **Statlog Status Classified as Pending Verification**:
   - *Decision*: Do not claim Statlog records are exact duplicates of Cleveland cases. Catalog official UCI description (modified 270-case numerical variant) and test exact duplicate overlap programmatically in Week 2 via row-hash matching.
4. **Framingham Educational Provenance Distinction**:
   - *Decision*: Clearly distinguish the 4,240-row teaching CSV (BioLINCC / teaching collections) from the full multi-generational NHLBI cohort, documenting educational usage rights and discrete tabular scope.
5. **TRIPOD and TRIPOD+AI Standards**:
   - *Decision*: Follow the 2015 TRIPOD statement and its 2024 machine learning expansion TRIPOD+AI ([REF-01b]) for transparent reporting across preprocessing, hyperparameter search, calibration, and subgroup performance.
6. **OOD Detection with Ledoit-Wolf Shrinkage & Documented Limitations**:
   - *Decision*: Use regularized Ledoit-Wolf shrinkage covariance on continuous features to prevent singular matrix inversion, and document the limitations of assuming multivariate Gaussianity and entropy overconfidence on shifted data.
7. **Validation on Small Sample Size ($N=303$)**:
   - *Decision*: Use Repeated Stratified $K$-Fold Cross-Validation with empirical bootstrap confidence intervals rather than noisy nested CV, explicitly documenting small sample uncertainty.
8. **ECG Scope Boundary**:
   - *Decision*: ECG is strictly limited to discrete tabular clinical features (`restecg`, `oldpeak`, `slope`). Waveform signal processing or file uploads are out of scope and deferred until after Week 2.

---

## 4. Academic Research Verification Status

20 peer-reviewed papers have been verified with complete bibliographic details, DOIs, and access statuses:
- **Clinical Prediction Standards**: Collins et al. (2015, *BMJ* [REF-01]), Collins et al. (2024, *BMJ* [REF-01b] TRIPOD+AI), Steyerberg & Vergouwe (2014, *Eur Heart J* [REF-02]).
- **Benchmark Datasets**: Detrano et al. (1989, *Am J Cardiol* [REF-03]), Wilson et al. (1998, *Circulation* [REF-04]).
- **Generalization & Validation**: Siontis et al. (2015, *J Clin Epidemiol* [REF-05]), Christodoulou et al. (2019, *J Clin Epidemiol* [REF-06]), Weng et al. (2017, *PLOS ONE* [REF-07]), Alaa et al. (2019, *PLOS ONE* [REF-08]).
- **Probability Calibration**: Van Calster et al. (2016, *J Clin Epidemiol* [REF-09]), Van Calster et al. (2019, *BMC Med* [REF-10]).
- **Explainable AI (SHAP)**: Lundberg & Lee (2017, *NeurIPS* [REF-11]), Lundberg et al. (2018, *Nature Biomed Eng* [REF-12]), Rudin (2019, *Nature Mach Intell* [REF-13]), Ward et al. (2020, *npj Digit Med* [REF-14]).
- **Fairness & Demographics**: Chen et al. (2021, *Annu Rev Biomed Data Sci* [REF-15]), Rajkomar et al. (2018, *Annals Intern Med* [REF-16]), Woodward (2019, *Int J Env Res Pub Health* [REF-17]).
- **Uncertainty & OOD Detection**: Ovadia et al. (2019, *NeurIPS* [REF-18]), Lee et al. (2018, *NeurIPS* [REF-19]).

*Verification Summary: 100% verified. Zero unverified or fabricated citations present.*

---

## 5. Dataset Status

- **UCI Heart Disease Collection (Cleveland, Hungarian, Long Beach, Switzerland)**:
  - Official UCI repository verified (Dataset ID: 45).
  - Four sub-cohort files documented; distinct missingness mechanisms, class distributions, and sample sizes documented separately.
  - Frozen Cleveland-only model derivation strategy established.
- **Statlog Heart Disease (Dataset ID: 145)**:
  - Official UCI description documented (270 records, 13 numerical attributes).
  - Row-hash programmatic duplicate matching scheduled for Week 2.
- **Framingham Heart Study Educational Subset**:
  - Educational 4,240-row teaching CSV distinguished from full NHLBI study.
  - 10-year incident hard CHD target verified; isolated into Pipeline B.

---

## 6. Current Issues, Scientific Audit Corrections & Outstanding Questions

### Audit Corrections Applied (Week 1 Scientific Audit):
1. **Statlog Provenance**: Marked exact duplication as a hypothesis pending Week 2 programmatic row-hash verification; documented official UCI description (270 instances, modified numeric version).
2. **Cross-Dataset Validation Protocol**: Specified Cleveland-only model development, preprocessing, calibrator fitting, and threshold selection; froze all parameters before zero-shot testing on external sites (Hungary, Switzerland, Long Beach). External data is never used for tuning.
3. **Dataset Compatibility**: Documented features, targets, missingness, and prevalence separately per site. Justified why forcing identical features across all sites is clinically and statistically inappropriate.
4. **Framingham Provenance**: Clearly distinguished the 4,240-row educational teaching CSV (BioLINCC / teaching archives) from the original multi-generational Framingham Heart Study cohort.
5. **Literature Accuracy & Qualified Fairness**: Verified exact publication details and qualified demographic fairness claims (mathematical impossibility of simultaneous parity under unequal prevalence). Added TRIPOD+AI (2024).
6. **OOD Methodology**: Documented limitations of Mahalanobis distance (Gaussian assumption) and prediction entropy; added regularized Ledoit-Wolf shrinkage covariance.
7. **Validation Methodology on Small Datasets**: Evaluated statistical instability of nested CV on $N=303$; adopted Repeated Stratified CV with bootstrap confidence intervals and documented uncertainty.
8. **Reporting Standards**: Incorporated TRIPOD+AI ([REF-01b]) alongside TRIPOD (2015).
9. **ECG Scope Boundary**: Formally restricted ECG data to discrete tabular features (`restecg`, `oldpeak`, `slope`); excluded waveform and signal processing.

### Outstanding Scientific Questions for Week 2:
- **Q1 (Statlog Duplication Concordance)**: Exactly how many of Statlog's 270 rows match Cleveland's 303 rows under feature tolerance matching, and how are missing values in Cleveland represented in Statlog?
- **Q2 (Non-Random Missingness Handling)**: In the Swiss dataset, cholesterol is recorded as 0 across all 123 rows (clinical omission, not true zero cholesterol). When evaluating the frozen Cleveland model zero-shot, how sensitive is model discrimination to imputing Cleveland's training median versus dropping cholesterol from the core bedside feature set?
- **Q3 (OOD Threshold Tuning)**: Does a 99th percentile $\chi^2$ cutoff on continuous features produce acceptable false-alarm rates on held-out Cleveland validation folds, or does it require empirical thresholding based on the natural distribution shift seen in Hungarian data?

---

## 7. Week 1 Acceptance Criteria Checklist

- [x] At least 10–15 relevant papers researched and verified (20 verified).
- [x] Literature review matrix created (`LITERATURE_REVIEW_MATRIX.md`).
- [x] Research gap is evidence-backed and cautiously worded (`RESEARCH_GAP.md`).
- [x] Research objectives (RO1–RO6) and questions (RQ1–RQ6) documented (`RESEARCH_OBJECTIVES.md`).
- [x] The two prediction experiments are clearly separated into isolated pipelines.
- [x] Dataset provenance and compatibility issues documented (`DATASET_DOCUMENTATION.md`).
- [x] Research methodology established with anti-leakage rules (`RESEARCH_METHODOLOGY.md`).
- [x] Architecture documented with modular monolith blueprints (`SYSTEM_ARCHITECTURE.md`).
- [x] Database design documented with PostgreSQL schema and ER diagram (`DATABASE_DESIGN.md`).
- [x] Requirements (`PROJECT_REQUIREMENTS.md`) and testing strategy (`TESTING_STRATEGY.md`) exist.
- [x] All 14 required project files created in project root.
- [x] The 10-week master roadmap exists (`MASTER_ROADMAP.md`).
- [x] `DEVELOPMENT_PROGRESS.md` maintained and updated.
- [x] No fabricated research, results, or unsupported claims are present.

---

## 8. Next Session Plan (Revised Week 2 Plan)

When Week 2 begins, the agent and developer will execute:
1. Ingest raw datasets into `data/raw/` (UCI Cleveland, Hungarian, Long Beach, Switzerland; Statlog; Framingham educational CSV).
2. Execute programmatic row-hash matching script between Statlog and Cleveland to resolve Question Q1.
3. Conduct automated Exploratory Data Analysis (EDA) scripts computing distributions, skewness, missingness maps, and correlation heatmaps per site.
4. Formulate the two feature configurations (13-feature Full Diagnostic Set vs. 10-feature Core Bedside Set) with statistical justification.
5. Author scikit-learn custom preprocessing transformers (`src/pipelines/preprocessors.py`) strictly isolating fold-level imputation and scaling.
6. Verify anti-leakage test assertions on mock splits.
