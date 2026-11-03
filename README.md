# HeartGuard AI: An Explainable, Calibrated and Cross-Dataset Machine Learning Framework for Heart Disease Risk Prediction

[![Python Version](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-green.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/Frontend-React%20%7C%20Vite%20%7C%20Tailwind-61DAFB.svg)](https://reactjs.org/)
[![PostgreSQL](https://img.shields.io/badge/Database-PostgreSQL%2016-336791.svg)](https://www.postgresql.org/)
[![Docker](https://img.shields.io/badge/Deployment-Docker%20Compose-2496ED.svg)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 1. Project Overview

**HeartGuard AI** is a 10-week final-year B.Tech major project that develops an explainable, calibrated, and externally validated machine learning framework for cardiovascular risk prediction. 

While existing applied machine learning literature frequently reports classification accuracies exceeding 85–90% on random train/test splits within single datasets (such as the UCI Cleveland dataset), these scores often represent internal overfitting and spectrum bias rather than real-world clinical reliability. When transferred to independent hospital cohorts, models frequently suffer severe performance degradation.

HeartGuard AI investigates and addresses five interconnected research pillars:
1. **Cross-Dataset Generalization**: Measuring external transfer performance across the four international centers of the UCI Heart Disease collection (Cleveland, Hungarian Institute of Cardiology, Long Beach VA, and University Hospital Switzerland).
2. **Probability Calibration**: Applying the Van Calster calibration hierarchy to transform raw heuristic outputs into reliable, empirical event probabilities via Platt scaling and Isotonic regression.
3. **Local & Global Explainability**: Leveraging cooperative game-theoretic SHAP (SHapley Additive exPlanations) to provide population-level beeswarm rankings and patient-level waterfall plots, paired with strict clinical warnings that associations do not imply biological causality.
4. **Demographic Fairness & Subgroup Equity**: Auditing disaggregated model performance across biological sex (Male vs. Female) and age brackets ($<50$, $50–65$, $>65$) to quantify and mitigate diagnostic disparities.
5. **Uncertainty & Out-of-Distribution (OOD) Guardrails**: Implementing multivariate Mahalanobis distance filters and predictive Shannon entropy thresholds to detect and reject anomalous or physiologically invalid patient inputs.

> [!WARNING]
> **Academic & Research Disclaimer**:  
> HeartGuard AI is strictly an **educational and scientific research prototype**, not a certified Software as a Medical Device (SaMD). Model outputs reflect statistical associations within retrospective benchmark cohorts and must not be used as clinical diagnostic determinations or prescriptive medical guidance.

---

## 2. Dual Clinical Pipeline Architecture

HeartGuard AI strictly segregates two clinically incompatible prediction tasks:

```
                       ┌───────────────────────────────────────────────┐
                       │          HeartGuard AI Data Systems           │
                       └──────────────────────┬────────────────────────┘
                                              │
              ┌───────────────────────────────┴───────────────────────────────┐
              ▼                                                               ▼
┌───────────────────────────────────────────┐   ┌───────────────────────────────────────────┐
│        EXPERIMENT PIPELINE A              │   │        EXPERIMENT PIPELINE B              │
│ Cross-Sectional Diagnostic Classification │   │ Longitudinal 10-Year Prognostic Risk     │
├───────────────────────────────────────────┤   ├───────────────────────────────────────────┤
│ Target: Anatomical Coronary Artery Stenosis│   │ Target: 10-Year Incident Hard CHD Event   │
│         (>= 50% diameter narrowing)       │   │         (Angina, MI, Fatal CHD)           │
│ Datasets:                                 │   │ Dataset:                                  │
│  - UCI Cleveland (Derivation / Baseline)  │   │  - Framingham Heart Study Cohort Subset   │
│  - UCI Hungarian (External Validation 1)  │   │    (4,240 derived educational dataset)    │
│  - UCI Long Beach VA (External Valid. 2)  │   │                                           │
│  - UCI Switzerland (External Valid. 3)    │   │                                           │
│  - Statlog Heart (Derived Verification)   │   │                                           │
└───────────────────────────────────────────┘   └───────────────────────────────────────────┘
```

---

## 3. Technology Stack

- **Frontend**: React 18, Vite, Tailwind CSS, Plotly.js (`react-plotly.js`), Lucide React, TanStack Query, React Hook Form, Zod.
- **Backend**: Python 3.11+, FastAPI, Pydantic v2, SQLAlchemy 2.0 (Asyncio), Alembic, Uvicorn.
- **Machine Learning**: Scikit-learn, XGBoost, LightGBM, CatBoost, SHAP, Imbalanced-learn, NumPy, Pandas, SciPy.
- **Database**: PostgreSQL 16.
- **Security**: Stateless JWT (HS256), Bcrypt password hashing.
- **DevOps & Containerization**: Docker, Docker Compose, Git.

---

## 4. Documentation Index

The repository maintains comprehensive academic and technical documentation:

| Document | Description |
| :--- | :--- |
| **[`MASTER_ROADMAP.md`](file:///c:/Users/USER/OneDrive/Desktop/HeartGuardAI/MASTER_ROADMAP.md)** | Comprehensive 10-week schedule, milestones, weekly WBS, and viva checkpoints. |
| **[`PROJECT_REQUIREMENTS.md`](file:///c:/Users/USER/OneDrive/Desktop/HeartGuardAI/PROJECT_REQUIREMENTS.md)** | ISO/IEC 29148 compliant functional, non-functional, security, and usability specifications. |
| **[`RESEARCH_GAP.md`](file:///c:/Users/USER/OneDrive/Desktop/HeartGuardAI/RESEARCH_GAP.md)** | Literature-backed empirical analysis establishing the 5 core research gaps. |
| **[`LITERATURE_REVIEW.md`](file:///c:/Users/USER/OneDrive/Desktop/HeartGuardAI/LITERATURE_REVIEW.md)** | In-depth synthesis of clinical prediction standards, algorithms, calibration, and explainability. |
| **[`LITERATURE_REVIEW_MATRIX.md`](file:///c:/Users/USER/OneDrive/Desktop/HeartGuardAI/LITERATURE_REVIEW_MATRIX.md)** | Structured comparative matrix of all 19 verified peer-reviewed publications. |
| **[`RESEARCH_OBJECTIVES.md`](file:///c:/Users/USER/OneDrive/Desktop/HeartGuardAI/RESEARCH_OBJECTIVES.md)** | Formal problem statement, Research Objectives (RO1–RO6), and Questions (RQ1–RQ6). |
| **[`RESEARCH_METHODOLOGY.md`](file:///c:/Users/USER/OneDrive/Desktop/HeartGuardAI/RESEARCH_METHODOLOGY.md)** | 10-model zoo, anti-leakage nested CV protocols, metrics, calibration, and SHAP pipelines. |
| **[`DATASET_DOCUMENTATION.md`](file:///c:/Users/USER/OneDrive/Desktop/HeartGuardAI/DATASET_DOCUMENTATION.md)** | Detailed provenance, schemas, class distributions, and compatibility matrix for all datasets. |
| **[`SYSTEM_ARCHITECTURE.md`](file:///c:/Users/USER/OneDrive/Desktop/HeartGuardAI/SYSTEM_ARCHITECTURE.md)** | Modular monolith design, component diagrams, data flows, and security architecture. |
| **[`DATABASE_DESIGN.md`](file:///c:/Users/USER/OneDrive/Desktop/HeartGuardAI/DATABASE_DESIGN.md)** | PostgreSQL 16 schema, ER diagrams, data dictionaries, constraints, and GIN indexes. |
| **[`DEVELOPMENT_PROGRESS.md`](file:///c:/Users/USER/OneDrive/Desktop/HeartGuardAI/DEVELOPMENT_PROGRESS.md)** | Live tracking ledger: completed tasks, blockers, decisions, and weekly acceptance criteria. |
| **[`TESTING_STRATEGY.md`](file:///c:/Users/USER/OneDrive/Desktop/HeartGuardAI/TESTING_STRATEGY.md)** | Multi-tier test pyramid: anti-leakage verification, calibration math, SHAP axioms, API tests. |
| **[`REFERENCES.md`](file:///c:/Users/USER/OneDrive/Desktop/HeartGuardAI/REFERENCES.md)** | Canonical bibliographic library of 19 verified academic citations with DOIs. |

---

## 5. Ten-Week Master Plan Summary

- **Week 1**: Research foundation, literature review, architecture, documentation. *(Current Week)*
- **Week 2**: Dataset acquisition, provenance checks, EDA, preprocessing, compatibility analysis.
- **Week 3**: Traditional ML model implementation and baseline experiments (5 models).
- **Week 4**: Advanced ML models, Optuna tuning, ensemble methods (5 models).
- **Week 5**: Cross-dataset evaluation, generalization experiments, statistical analysis.
- **Week 6**: SHAP, calibration, fairness, uncertainty and OOD evaluation.
- **Week 7**: FastAPI backend, database, authentication, prediction APIs and tests.
- **Week 8**: React frontend and interactive research dashboard.
- **Week 9**: Integration, testing, Docker deployment, security and bug fixing.
- **Week 10**: Final report, research paper draft, diagrams, PPT, screenshots and viva preparation.

---

## 6. Directory Structure (Target Scaffold)

```
HeartGuardAI/
├── .gitignore
├── README.md
├── MASTER_ROADMAP.md
├── PROJECT_REQUIREMENTS.md
├── RESEARCH_GAP.md
├── LITERATURE_REVIEW.md
├── LITERATURE_REVIEW_MATRIX.md
├── RESEARCH_OBJECTIVES.md
├── RESEARCH_METHODOLOGY.md
├── DATASET_DOCUMENTATION.md
├── SYSTEM_ARCHITECTURE.md
├── DATABASE_DESIGN.md
├── DEVELOPMENT_PROGRESS.md
├── TESTING_STRATEGY.md
├── REFERENCES.md
├── data/                    # Cleaned and raw datasets (Week 2)
│   ├── raw/
│   └── processed/
├── src/                     # Backend application & ML source code (Weeks 3-7)
│   ├── api/                 # FastAPI routes and middleware
│   ├── core/                # Configuration and security settings
│   ├── db/                  # Database models and migrations
│   ├── ml/                  # Model zoo, calibration, SHAP, and OOD engines
│   └── pipelines/           # Preprocessors and data loaders
├── tests/                   # Automated test suites (unit, integration, ML integrity)
└── frontend/                # React Vite Tailwind client application (Week 8)
```
