# HeartGuard AI: Development Progress & Project Ledger

**Current Status**: Week 1 Complete  
**Current Milestone**: Milestone 1 — Research Foundation, Literature Verification, System Design & Architecture  
**Developer Commitment**: 6–8 hours/week (1 flexible working day per week)  
**Last Updated**: November 2026 (End of Week 1 Execution)

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
| **TASK 2** | Research and verify 10–15 peer-reviewed academic papers | **COMPLETED** | 19 papers independently verified via PubMed / DOIs |
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
   - *Decision*: Strictly isolate Experiment Pipeline A (Diagnostic CAD classification on UCI multicenter collection) from Experiment Pipeline B (Prognostic 10-year incident CHD risk on Framingham cohort).
   - *Rationale*: Merging or cross-mapping these datasets creates an invalid hybrid task conflating cross-sectional angiographic stenosis with longitudinal prospective event incidence.
2. **Modular Monolith over Microservices**:
   - *Decision*: Build a clean modular monolith backend using FastAPI, Pydantic, and SQLAlchemy within a single unified containerized environment.
   - *Rationale*: Eliminates unnecessary network serialization overhead and distributed transaction complexity while preserving strict internal boundaries.
3. **TRIPOD Guidelines as Scientific Anchor**:
   - *Decision*: Follow the 22-item TRIPOD checklist across all model development and validation reporting.
   - *Rationale*: Ensures publication-grade reporting transparency and guarantees high academic credibility during viva defense.
4. **Comprehensive Evaluation Metric Standard**:
   - *Decision*: Bar classification accuracy from being the primary model selection criterion. Models will be selected using ROC-AUC, PR-AUC, Sensitivity at fixed Specificity, and Brier calibration score.
   - *Rationale*: Mitigates deceptive performance claims caused by class imbalance and miscalibrated probabilities.
5. **Non-Causal Interpretability Disclaimer**:
   - *Decision*: Every SHAP visualization in the user interface and API response must include an explicit disclaimer stating that feature attributions represent algorithmic correlations, not clinical causality.
   - *Rationale*: Adheres to high-stakes decision recommendations (Rudin, 2019) and prevents clinician misinterpretation.

---

## 4. Academic Research Verification Status

19 peer-reviewed papers have been verified with complete bibliographic details, DOIs, and access statuses:
- **Clinical Prediction Standards**: Collins et al. (2015, *BMJ* [REF-01]), Steyerberg & Vergouwe (2014, *Eur Heart J* [REF-02]).
- **Benchmark Datasets**: Detrano et al. (1989, *Am J Cardiol* [REF-03]), Wilson et al. (1998, *Circulation* [REF-04]).
- **Generalization & Validation**: Siontis et al. (2015, *J Clin Epidemiol* [REF-05]), Christodoulou et al. (2019, *J Clin Epidemiol* [REF-06]), Weng et al. (2017, *PLOS ONE* [REF-07]), Alaa et al. (2019, *PLOS ONE* [REF-08]).
- **Probability Calibration**: Van Calster et al. (2016, *J Clin Epidemiol* [REF-09]), Van Calster et al. (2019, *BMC Med* [REF-10]).
- **Explainable AI (SHAP)**: Lundberg & Lee (2017, *NeurIPS* [REF-11]), Lundberg et al. (2018, *Nature Biomed Eng* [REF-12]), Rudin (2019, *Nature Mach Intell* [REF-13]), Ward et al. (2020, *npj Digit Med* [REF-14]).
- **Fairness & Demographics**: Chen et al. (2021, *Annu Rev Biomed Data Sci* [REF-15]), Rajkomar et al. (2018, *Annals Intern Med* [REF-16]), Woodward (2019, *Int J Environ Res Pub Health* [REF-17]).
- **Uncertainty & OOD Detection**: Ovadia et al. (2019, *NeurIPS* [REF-18]), Lee et al. (2018, *NeurIPS* [REF-19]).

*Verification Summary: 100% verified. Zero unverified or fabricated citations present.*

---

## 5. Dataset Status

- **UCI Heart Disease Collection (Cleveland, Hungarian, Long Beach, Switzerland)**:
  - Official UCI repository verified (Dataset ID: 45).
  - Four sub-cohort files documented; feature schema mapped; missingness patterns characterized.
  - Ready for ingestion and programmatic cleaning in Week 2.
- **Statlog Heart Disease (Dataset ID: 145)**:
  - Repository verified; confirmed to be a derivative of Cleveland complete cases.
  - Marked for row-hashing duplicate verification in Week 2.
- **Framingham Heart Study (4,240-row Educational Subset)**:
  - 10-year incident hard CHD target verified; 15 features mapped; severe class imbalance ($\sim 15\%$ event prevalence) documented.
  - Isolated into Pipeline B.

---

## 6. Current Issues and Blockers

- **Zero Current Blockers**: All Week 1 research foundation and specification tasks completed successfully.
- **Identified Challenges for Week 2**:
  - Substantial missingness in the European and Long Beach UCI files (`chol` recorded as 0 in Hungary and Switzerland; `ca` and `thal` absent in $\sim 90\%$ of Hungarian records).
  - Handling strategy: Implement a harmonized core feature subset for cross-dataset evaluation that avoids leaking labels or imputing non-random missingness unsafely.

---

## 7. Week 1 Acceptance Criteria Checklist

- [x] At least 10–15 relevant papers researched and verified (19 verified).
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

## 8. Next Session Plan (Week 2 Kickoff)

When Week 2 begins, the agent and developer will execute:
1. Create directory structure: `data/raw/`, `data/processed/`, `src/pipelines/`, `notebooks/`.
2. Programmatically download and ingest the UCI 4-center raw datasets and Framingham CSV.
3. Perform exact duplicate-row hashing between Statlog Heart and Cleveland dataset.
4. Conduct automated Exploratory Data Analysis (EDA) scripts computing distributions, skewness, correlations, and missing value maps.
5. Author scikit-learn custom preprocessing transformers enforcing strict fold-level imputation and scaling isolation.
6. Verify anti-leakage test assertions on sample splits.
