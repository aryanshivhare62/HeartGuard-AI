# HeartGuard AI: Project Requirements Specification (PRS)

**Project Title**: HeartGuard AI: An Explainable, Calibrated and Cross-Dataset Machine Learning Framework for Heart Disease Risk Prediction  
**Standard**: ISO/IEC/IEEE 29148:2018; TRIPOD Statement (2015) & TRIPOD+AI (2024 [REF-01b])  
**Date**: November 2026 (Week 1 Phase)

---

## 1. Project Purpose & Scope

### 1.1 Purpose
HeartGuard AI is an end-to-end academic research framework and demonstration decision-support web application. It addresses the lack of independent external validation, probability calibration, interpretable explanations, and demographic fairness in machine learning models for cardiovascular risk assessment.

### 1.2 Intended Audience
- **Academic Evaluators & Viva Examination Panels**: Reviewing research rigor, methodology, and empirical evidence.
- **Data Scientists & ML Engineers**: Inspecting pipeline reproducibility, anti-leakage architecture, and benchmarking protocols.
- **Clinical Researchers**: Interacting with transparent model explanations, calibration curves, and subgroup disparity analytics.

### 1.3 Regulatory Boundary & Non-Claims
> [!WARNING]
> HeartGuard AI is **NOT** a certified Medical Device or Software as a Medical Device (SaMD) under FDA (21 CFR Part 820) or EU MDR (Regulation 2017/745). Model outputs represent statistical associations within benchmark cohorts and must never be interpreted as certified medical diagnoses or prescriptive treatment mandates.

---

## 2. Functional Requirements (FR)

### Module 1: Authentication & Access Control
- **FR-01**: The system shall support user registration with email, password (minimum 8 characters, complexity check), and full name.
- **FR-02**: The system shall authenticate users via OAuth2 Password flow and issue signed JSON Web Tokens (JWT) with configurable expiration (default: 60 minutes).
- **FR-03**: The system shall enforce role-based authorization differentiating `CLINICIAN` (prediction workflows, patient profiles) and `RESEARCHER` (benchmark comparisons, cross-dataset evaluations, retraining triggers).

### Module 2: Dual Prediction Pipelines
- **FR-04 (Diagnostic Pipeline A)**: The system shall provide an inference interface accepting the 13 clinical predictors of the UCI Heart Disease protocol and outputting the probability of $\ge 50\%$ angiographic coronary artery stenosis.
- **FR-05 (Prognostic Pipeline B)**: The system shall provide a separate, strictly isolated inference interface accepting Framingham risk variables and predicting the 10-year probability of incident hard coronary heart disease events.
- **FR-06 (Strict Segregation Enforced)**: The system shall prevent the combination, joint inference, or cross-feature mapping of Pipeline A and Pipeline B data models.

### Module 3: Probability Calibration & Reliability
- **FR-07**: For every inference request, the system shall compute and return both uncalibrated raw model probabilities and post-hoc calibrated probabilities (Platt Sigmoid and Isotonic).
- **FR-08**: The system shall visualize interactive calibration curves (10-bin deciles and loess smoothing) comparing predicted risk versus observed event rates.
- **FR-09**: The system shall display Brier score, calibration slope, and calibration-in-the-large intercept for each benchmarked model.

### Module 4: Explainability & Interpretability (SHAP)
- **FR-10**: For every individual patient prediction, the system shall generate local SHAP attributions and render an interactive waterfall plot displaying positive and negative risk contributors.
- **FR-11**: The system shall render global SHAP summary beeswarm plots and mean absolute feature importance rankings across all benchmarked models.
- **FR-12**: All explanation views shall display an explicit disclaimer that feature attributions represent model statistical correlations, not biological causation.

### Module 5: Cross-Dataset Benchmarking & Generalization Analytics
- **FR-13**: The system shall provide an interactive Model Comparison Dashboard benchmarking all 10 algorithms across discrimination, precision-recall, and calibration metrics.
- **FR-14**: The system shall display a dedicated Cross-Dataset Generalization Matrix showing performance drops ($\Delta\text{AUC}$, $\Delta\text{Brier}$) when models trained on Cleveland are evaluated zero-shot on Hungary, Long Beach, and Switzerland.

### Module 6: Demographic Fairness & Subgroup Auditing
- **FR-15**: The system shall compute and display disaggregated performance metrics stratified by Biological Sex (Male vs. Female) and Age Categories ($<50$, $50–65$, $>65$).
- **FR-16**: The system shall report Equalized Odds differentials ($|\Delta\text{TPR}| + |\Delta\text{FPR}|$) to quantify potential demographic disparities.

### Module 7: Uncertainty & Out-of-Distribution (OOD) Guardrails
- **FR-17**: The system shall validate all clinical inputs against hard physiological sanity boundaries, rejecting out-of-range values with informative 422 Unprocessable Entity errors.
- **FR-18**: Prior to model inference, the system shall compute the multivariate Mahalanobis distance across continuous features using a regularized Ledoit-Wolf shrinkage covariance matrix fit on the Cleveland training cohort.
- **FR-19**: When the Mahalanobis distance exceeds the 99th percentile $\chi^2$ threshold or prediction entropy exceeds $0.95$ bits, the system shall attach a prominent "High Uncertainty / Out-of-Distribution" warning to the prediction response, documenting known limitations regarding non-Gaussian features and entropy overconfidence.

### Module 8: Audit Logging & History
- **FR-20**: The system shall persist every prediction request, sanitized input vector, output probabilities, calibration method, OOD flag, and SHAP vector into the PostgreSQL `prediction_logs` table.
- **FR-21**: Authenticated clinicians shall be able to view, search, and export their past prediction audit logs.

### Module 9: Input Modality & ECG Scope Boundary
- **FR-22 (ECG Tabular Modality Boundary)**: The system shall process ECG data strictly as discrete tabular clinical observations (`restecg`: categorical 0/1/2; `oldpeak`: continuous ST-depression in mm; `slope`: categorical 1/2/3). The system shall **NOT** accept ECG waveform file uploads, digital raw signal time-series, or execute deep 1D/2D CNN waveform classification. Any expansion of ECG modality is formally deferred until after Week 2 data exploration.

---

## 3. Non-Functional Requirements (NFR)

### 3.1 Performance & Latency
- **NFR-01 (Inference Latency)**: End-to-end prediction execution (including Pydantic validation, pipeline preprocessing, calibrated inference, and TreeSHAP calculation) shall complete in $< 350$ milliseconds for a single patient query.
- **NFR-02 (Frontend Performance)**: Initial frontend bundle load time shall be $< 2.0$ seconds on standard broadband, with Google Lighthouse Performance score $\ge 90$.

### 3.2 Security & Data Privacy
- **NFR-03 (Password Hashing)**: User passwords must be hashed using bcrypt with a minimum work factor of 12.
- **NFR-04 (Token Security)**: JWT tokens must use HMAC-SHA256 with secrets configured via external environment variables; tokens must expire within 60 minutes.
- **NFR-05 (De-Identification)**: No direct Personally Identifiable Information (PII) such as patient government ID or real names shall be stored in the database.

### 3.3 Reliability & Reproducibility
- **NFR-06 (Deterministic Training)**: All random seeds (`numpy.random.seed`, `random_state`) across cross-validation splits, model fitting, and hyperparameter tuning must be pinned to ensure 100% reproducible benchmark scores.
- **NFR-07 (Availability)**: The backend API shall implement a health check endpoint (`GET /health`) returning container status, database connection state, and loaded model registry status.

### 3.4 Usability & Accessibility
- **NFR-08**: The web interface must be fully responsive, supporting screen widths from $375\text{px}$ (mobile) to $1920\text{px}$ (desktop).
- **NFR-09**: Visualizations must comply with WCAG 2.1 AA contrast standards, utilizing accessible color palettes for colorblind users.

---

## 4. Software Engineering & Architecture Requirements

- **SR-01 (Modular Monolith)**: The backend codebase must maintain clean physical separation across `api/`, `core/`, `models/`, `pipelines/`, `schemas/`, and `services/`.
- **SR-02 (Type Safety)**: Python code must be statically typed using type hints; frontend must use TypeScript or strictly structured JSDoc Zod validation.
- **SR-03 (Automated Testing)**: The codebase must maintain $\ge 85\%$ test coverage across data transformation, model inference, calibration, and API endpoints.
- **SR-04 (Containerization)**: The complete system must build and run cleanly on any host OS running Docker Compose using a single command: `docker-compose up --build`.

