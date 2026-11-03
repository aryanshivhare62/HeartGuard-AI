# HeartGuard AI: Master 10-Week Engineering & Research Roadmap

**Project**: HeartGuard AI: An Explainable, Calibrated and Cross-Dataset Machine Learning Framework for Heart Disease Risk Prediction  
**Allocation**: 10 Weeks, 6–8 Hours/Week (1 Flexible Working Day per Week)  
**Methodological Cadence**: Agile-Research Hybrid with Strict Weekly Milestones and Viva Checkpoints

---

## 1. Master Schedule Overview

```
Week 01: [██████████] Research Foundation, Literature Verification, System Design, Documentation
Week 02: [          ] Dataset Ingestion, Provenance Auditing, Exploratory Data Analysis, Anti-Leakage Preprocessing
Week 03: [          ] Traditional Machine Learning Baselines & Cross-Validation Benchmarks (5 Models)
Week 04: [          ] Advanced Gradient Boosted Ensembles & Hyperparameter Optimization (5 Models)
Week 05: [          ] Cross-Dataset Generalization Experiments & External Validation Transfer Analysis
Week 06: [          ] Explainability (SHAP), Probability Calibration, Subgroup Fairness, & OOD Detection
Week 07: [          ] FastAPI Backend Services, PostgreSQL Schema, JWT Security, & API Integration Tests
Week 08: [          ] React + Vite + Tailwind Frontend, Interactive Plotly Visualizations, Clinical UI
Week 09: [          ] End-to-End System Integration, Docker Orchestration, Security Audit, & Hardening
Week 10: [          ] Final Project Thesis, Academic Research Paper Draft, Presentation Deck, & Viva Defense Prep
```

---

## 2. Weekly Work Breakdown Structure (WBS)

### Week 1: Research Foundation, Literature Review & Architecture
- **Primary Focus**: Establish academic rigor, literature verification, research questions, and system blueprints.
- **Tasks**:
  1. Inspect workspace and configure repository root.
  2. Verify 15+ high-impact academic papers across PubMed, BMJ, IEEE, Springer, Nature, and Crossref.
  3. Author `LITERATURE_REVIEW.md`, `LITERATURE_REVIEW_MATRIX.md`, and `REFERENCES.md`.
  4. Formulate evidence-backed `RESEARCH_GAP.md`, `RESEARCH_OBJECTIVES.md`, and `RESEARCH_METHODOLOGY.md`.
  5. Architect modular monolith blueprints in `SYSTEM_ARCHITECTURE.md`, `DATABASE_DESIGN.md`, and `PROJECT_REQUIREMENTS.md`.
  6. Formulate comprehensive `TESTING_STRATEGY.md` and initialize Git tracking.
- **Deliverables**: 14 foundational specification files in project root; verified reference library.
- **Viva Checkpoint**: Ability to articulate the 5 research pillars, the 4-level calibration hierarchy, and why random-split validation inflates performance.

---

### Week 2: Dataset Acquisition, Provenance Checks & Preprocessing
- **Primary Focus**: Source data ingestion, strict schema enforcement, missing value characterization, and feature compatibility analysis.
- **Tasks**:
  1. Ingest official UCI 4-center raw files (`processed.cleveland.data`, `processed.hungarian.data`, `processed.switzerland.data`, `processed.va.data`).
  2. Ingest Statlog Heart Disease dataset and execute duplicate-row hash matching against Cleveland.
  3. Ingest Framingham 4,240-record cohort dataset into isolated Pipeline B directory.
  4. Conduct exploratory data analysis (EDA): distributions, correlation heatmaps, missingness profiles, and class imbalance.
  5. Build scikit-learn anti-leakage preprocessing transformers (`src/pipelines/preprocessors.py`).
- **Deliverables**: Cleaned data artifacts, EDA exploratory notebooks/scripts, automated data loading pipelines with Pydantic validation.
- **Viva Checkpoint**: Explaining why Cleveland and Framingham cannot be merged, and proving how imputation prevents test-set leakage.

---

### Week 3: Traditional ML Baselines & Cross-Validation Benchmarks
- **Primary Focus**: Build, tune, and evaluate 5 traditional/interpretable machine learning algorithms.
- **Tasks**:
  1. Implement $L_2$-regularized Logistic Regression pipeline with balanced class weighting.
  2. Implement Decision Tree (CART) classifier with tree-depth pruning.
  3. Implement Random Forest Classifier with out-of-bag scoring.
  4. Implement K-Nearest Neighbors (KNN) with distance-weighted voting.
  5. Implement Support Vector Classifier (SVC) with RBF kernel and probability Platt estimation.
  6. Execute Repeated Stratified 5-Fold Cross-Validation on Cleveland dataset.
  7. Generate baseline benchmark tables and confusion matrices.
- **Deliverables**: Serialized baseline model artifacts (`.joblib`), cross-validation benchmark leaderboard, baseline performance report.
- **Viva Checkpoint**: Comparing logistic regression odds ratios with tree splits, and defending why accuracy is an inadequate clinical metric.

---

### Week 4: Advanced ML Models, Tuning & Ensemble Methods
- **Primary Focus**: Implement state-of-the-art gradient boosted decision trees and meta-ensembles.
- **Tasks**:
  1. Implement HistGradientBoosting / Gradient Boosting Classifier.
  2. Implement XGBoost Classifier with column subsampling and early stopping.
  3. Implement LightGBM Classifier with leaf-wise histogram binning.
  4. Implement CatBoost Classifier with ordered categorical handling.
  5. Implement Stacking Classifier combining top diverse models using logistic regression meta-learner.
  6. Execute systematic hyperparameter optimization via `Optuna` inside cross-validation loops.
  7. Compare class weighting vs. SMOTE oversampling applied inside training folds.
- **Deliverables**: Tuned ensemble model artifacts, Optuna optimization logs, comprehensive 10-model comparison matrix.
- **Viva Checkpoint**: Explaining gradient boosting mechanics, regularization differences between XGBoost and LightGBM, and SMOTE fold isolation.

---

### Week 5: Cross-Dataset Evaluation & Generalization Experiments
- **Primary Focus**: The central research investigation—evaluating external generalization across independent cohorts.
- **Tasks**:
  1. Train models on primary Cleveland dataset ($N=303$) with harmonized feature subset.
  2. Perform zero-shot external evaluation on Hungarian cohort ($N=294$).
  3. Perform zero-shot external evaluation on Long Beach VA cohort ($N=200$).
  4. Perform zero-shot external evaluation on Switzerland cohort ($N=123$).
  5. Quantify generalization degradation ($\Delta\text{ROC-AUC}$, $\Delta\text{PR-AUC}$, $\Delta\text{Brier}$).
  6. Execute Delong test and non-parametric bootstrap tests for statistical significance of performance drops.
  7. Evaluate whether complex ensembles or regularized linear models exhibit superior generalization resilience.
- **Deliverables**: Empirical Cross-Dataset Generalization Report, degradation comparison heatmaps, statistical significance tables.
- **Viva Checkpoint**: Defending empirical findings on why external discrimination drops and how covariate shift impacts models.

---

### Week 6: SHAP, Calibration, Fairness & OOD Detection
- **Primary Focus**: The four pillars of trustworthy ML: interpretability, calibration, equity, and safety.
- **Tasks**:
  1. Implement TreeSHAP and KernelSHAP explanation engines (`src/explainability/`).
  2. Generate global summary beeswarm plots, feature dependence plots, and local individual waterfall plots.
  3. Implement probability calibration pipelines: Platt Scaling (Sigmoid) and Isotonic Regression.
  4. Compute Brier scores, calibration slopes, calibration intercepts, and plot 10-bin calibration curves.
  5. Implement demographic fairness auditing disaggregating metrics by Sex (Male/Female) and Age ($<50$, $50–65$, $>65$).
  6. Implement multivariate Mahalanobis distance out-of-distribution (OOD) filter and prediction entropy checks.
- **Deliverables**: Calibrated model wrappers, SHAP attribution generators, fairness disparity audit reports, OOD filter module.
- **Viva Checkpoint**: Explaining why SHAP does not prove biological causality, how Platt scaling differs from isotonic regression, and the mathematics of Mahalanobis distance.

---

### Week 7: FastAPI Backend, Database & Prediction APIs
- **Primary Focus**: Build robust, production-grade REST APIs, relational persistence, and authentication.
- **Tasks**:
  1. Configure PostgreSQL 16 database connection using SQLAlchemy 2.0 (asyncio).
  2. Write Alembic database migrations for users, patient profiles, models, benchmarks, and prediction logs.
  3. Implement user registration, password hashing (bcrypt), and JWT authentication routes.
  4. Implement `POST /api/v1/predict/diagnostic` with Pydantic validation, OOD checking, calibrated inference, and TreeSHAP computation.
  5. Implement `POST /api/v1/predict/prognostic` for Framingham 10-year risk assessment.
  6. Implement `GET /api/v1/benchmarks` and `GET /api/v1/benchmarks/cross-dataset` endpoints.
  7. Author comprehensive test suite using `pytest` and `httpx` achieving $\ge 85\%$ coverage.
- **Deliverables**: Fully operational FastAPI application, migration scripts, automated API test suite passing 100%.
- **Viva Checkpoint**: Code walkthrough of asynchronous API design, Pydantic schema validation, and secure JWT verification.

---

### Week 8: React Frontend & Interactive Research Dashboard
- **Primary Focus**: Build responsive, intuitive clinical decision-support and research evaluation UI.
- **Tasks**:
  1. Initialize React + Vite + Tailwind CSS project with component library.
  2. Build User Authentication UI (Login, Registration, Protected Routes).
  3. Build Clinical Assessment Form with real-time range validation (Zod) and physiological error badges.
  4. Build Interactive Prediction Results View: Calibrated probability gauge, uncertainty warning banner, and interactive Plotly SHAP waterfall chart.
  5. Build Model Benchmark Dashboard: Interactive ROC/PR curves, calibration decile plots, and 10-model comparison leaderboard.
  6. Build Cross-Dataset Generalization Explorer showing degradation matrices across the 4 international centers.
  7. Build Fairness & Subgroup Auditor visualizer displaying disaggregated sensitivity/specificity bars.
- **Deliverables**: Polished React web application, interactive Plotly visualization components, responsive clinical workflows.
- **Viva Checkpoint**: Live demonstration of patient prediction workflow, SHAP explanation breakdown, and interactive benchmark dashboard.

---

### Week 9: Integration, Docker Deployment, Security & Hardening
- **Primary Focus**: End-to-end integration, container orchestration, performance profiling, and security hardening.
- **Tasks**:
  1. Author multi-stage production Dockerfiles for React frontend and FastAPI backend.
  2. Configure `docker-compose.yml` linking frontend, backend, and PostgreSQL with persistent volumes and health checks.
  3. Implement CORS security headers, rate limiting middleware, and payload size bounds.
  4. Execute full end-to-end integration tests (`Playwright` / API test harness).
  5. Measure inference latency; optimize model loading via warm-start caching.
  6. Conduct comprehensive code cleanup, docstring verification, and linting (`ruff`, `black`, `mypy`).
- **Deliverables**: Single-command runnable Docker Compose deployment, integration test reports, security audit sign-off.
- **Viva Checkpoint**: Demonstrating seamless one-command deployment (`docker-compose up --build`) and explaining container networking.

---

### Week 10: Final Thesis, Research Paper, Presentation & Viva Prep
- **Primary Focus**: Academic synthesis, thesis documentation, paper publication draft, and examination readiness.
- **Tasks**:
  1. Author the Final B.Tech Major Project Thesis following institutional formatting guidelines.
  2. Draft an IEEE/Springer format research conference paper summarizing the cross-dataset and calibration findings.
  3. Create professional PowerPoint presentation deck (20–25 slides) with high-resolution system diagrams and result charts.
  4. Create comprehensive Viva Examination Question & Answer Preparation Guide covering all mathematical and clinical concepts.
  5. Record high-resolution application screenshots and create an end-to-end video walkthrough.
  6. Tag the canonical Git release `v1.0.0` and finalize project documentation.
- **Deliverables**: Final Major Project Thesis (PDF), Research Paper Manuscript (PDF/LaTeX), Viva PPT Deck, Comprehensive Viva FAQ Guide.
- **Viva Checkpoint**: Final mock viva simulation; defending all research objectives, empirical findings, and engineering choices.

