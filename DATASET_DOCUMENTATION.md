# HeartGuard AI: Dataset Documentation & Compatibility Analysis

**Author**: Principal ML Research Engineer & Data Scientist  
**Date**: November 2026 (Week 1 Research Phase)  
**Strict Compliance**: TRIPOD Guidelines (Item 4: Source of data, Item 6: Predictors, Item 7: Outcome)

---

## 1. Architectural Dataset Segregation Principle

A primary flaw in contemporary heart disease machine learning tutorials and published papers is the uncritical pooling of datasets with radically different clinical definitions, study populations, and prediction horizons. 

HeartGuard AI establishes an uncompromising **two-pipeline data segregation architecture**:

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
│ Population: Symptomatic patients referred │   │ Population: Asymptomatic community cohort │
│             for coronary angiography      │   │             free of CVD at baseline       │
│ Primary Datasets:                         │   │ Primary Dataset:                          │
│  - UCI Cleveland (Derivation / Baseline)  │   │  - Framingham Heart Study Cohort Subset   │
│  - UCI Hungarian (External Validation 1)  │   │    (4,240 derived educational dataset)    │
│  - UCI Long Beach VA (External Valid. 2)  │   │                                           │
│  - UCI Switzerland (External Valid. 3)    │   │                                           │
│  - Statlog Heart (Derived Verification)   │   │                                           │
└───────────────────────────────────────────┘   └───────────────────────────────────────────┘
```

> [!CAUTION]
> **Strict Rule on Cross-Pipeline Incompatibility**:  
> Features, labels, and records from Pipeline A (UCI Diagnostic) and Pipeline B (Framingham Prognostic) **MUST NEVER** be merged, concatenated, or trained together. They represent fundamentally distinct clinical tasks.

---

## 2. Pipeline A: Diagnostic Datasets (UCI Multicenter Collection)

### 2.1 The UCI Heart Disease Repository (Dataset ID: 45)
- **Official Source**: UC Irvine Machine Learning Repository  
  URL: [https://archive.ics.uci.edu/dataset/45/heart+disease](https://archive.ics.uci.edu/dataset/45/heart+disease)
- **Donation Date**: July 1988 by David W. Aha
- **Primary Clinical Citation**: Detrano et al., *The American Journal of Cardiology*, 1989 [REF-03]
- **Original Scope**: 76 raw clinical attributes collected across four geographic clinical centers.
- **Standardized Machine Learning Subset**: 14 standard attributes (13 clinical predictors + 1 target variable `num`).
- **Target Variable Definition**: `num` indicates the angiographic presence of coronary heart disease:
  - `0`: $<50\%$ diameter narrowing in all major coronary vessels (Absence of significant CAD).
  - `1`, `2`, `3`, `4`: $\ge 50\%$ diameter narrowing in 1, 2, 3, or 4 major coronary arteries.
  - Standard Binary Task: $\text{Target} = \mathbb{I}(\text{num} > 0)$.
- **License**: Creative Commons Attribution 4.0 International (CC BY 4.0).

---

### 2.2 Detailed Sub-Center Characteristics

#### 1. Cleveland Clinic Foundation (Ohio, USA) — Primary Derivation Cohort
- **File**: `processed.cleveland.data`
- **Total Records**: 303 patient cases.
- **Target Prevalence**: 139 positive ($\ge 50\%$ stenosis, 45.87%), 164 negative (54.13%).
- **Demographics**: Mean age $54.4 \pm 9.0$ years; 206 males (68.0%), 97 females (32.0%).
- **Missing Values**: Total 6 values across two features:
  - `ca` (number of major vessels colored by fluoroscopy): 4 missing values (`?`).
  - `thal` (thallium heart scan): 2 missing values (`?`).
- **Clinical Setting**: Patients referred for diagnostic coronary arteriography due to suspected angina pectoris.
- **HeartGuard Role**: Primary training, internal cross-validation, and algorithm benchmarking derivation set.

#### 2. Hungarian Institute of Cardiology (Budapest) — External Test Set 1
- **File**: `processed.hungarian.data`
- **Total Records**: 294 patient cases.
- **Target Prevalence**: 106 positive (36.05%), 188 negative (63.95%).
- **Demographics**: Mean age $47.9 \pm 7.8$ years; 213 males (72.4%), 81 females (27.6%).
- **Missing Values**:
  - `trestbps` (resting blood pressure): 1 missing.
  - `chol` (serum cholesterol): 23 missing (frequently coded as 0).
  - `fbs` (fasting blood sugar): 8 missing.
  - `restecg` (resting ECG): 1 missing.
  - `slope`: 190 missing.
  - `ca`: 291 missing ($\sim 99\%$ missingness; fluoroscopy rarely performed).
  - `thal`: 266 missing ($\sim 90\%$ missingness; thallium scans rarely performed).
- **HeartGuard Role**: External validation of models trained on compatible feature subsets (excluding or imputing `ca`, `thal`, `slope`).

#### 3. University Hospital Zurich & Basel (Switzerland) — External Test Set 2
- **File**: `processed.switzerland.data`
- **Total Records**: 123 patient cases.
- **Target Prevalence**: 115 positive (93.50%—heavily skewed severe disease population), 8 negative (6.50%).
- **Demographics**: Mean age $55.3 \pm 9.1$ years; 113 males (91.9%), 10 females (8.1%).
- **Missing Values**: Extensive missingness in `chol` (all recorded as 0), `fbs`, `ca`, `thal`, and exercise parameters.
- **HeartGuard Role**: Extreme distribution shift stress test. Evaluates how models handle acute spectrum bias (inpatient catheterization with near-universal severe CAD).

#### 4. V.A. Medical Center (Long Beach, California) — External Test Set 3
- **File**: `processed.va.data`
- **Total Records**: 200 patient cases.
- **Target Prevalence**: 151 positive (75.50%), 49 negative (24.50%).
- **Demographics**: Veterans Affairs male population: mean age $59.4 \pm 7.9$ years; 194 males (97.0%), 6 females (3.0%).
- **Missing Values**: Severe missingness in `chol` (frequent 0s), `ca`, and `thal`.
- **HeartGuard Role**: External validation under extreme male demographic bias and high disease severity.

---

### 2.3 The 14 Standard UCI Features and Canonical Schema

| Field Name | Variable Name | Data Type | Range / Domain | Clinical Description & Diagnostic Meaning |
| :--- | :--- | :--- | :--- | :--- |
| 1 | `age` | Integer | $29 – 77$ | Patient chronological age in years |
| 2 | `sex` | Binary | `1`: Male; `0`: Female | Biological sex |
| 3 | `cp` | Categorical | `1`: Typical angina<br>`2`: Atypical angina<br>`3`: Non-anginal pain<br>`4`: Asymptomatic | Chest pain type according to clinical presentation |
| 4 | `trestbps` | Continuous | $94 – 200$ mmHg | Resting systolic blood pressure on admission to hospital |
| 5 | `chol` | Continuous | $126 – 564$ mg/dL | Serum cholesterol in mg/dL (Note: 0 indicates missing in Hungarian/VA) |
| 6 | `fbs` | Binary | `1`: $>120$ mg/dL;<br>`0`: $\le 120$ mg/dL | Fasting blood sugar elevated status (diabetic indicator) |
| 7 | `restecg` | Categorical | `0`: Normal<br>`1`: ST-T wave abnormality<br>`2`: Left ventricular hypertrophy | Resting electrocardiographic results (Estes' criteria) |
| 8 | `thalach` | Continuous | $71 – 202$ bpm | Maximum heart rate achieved during treadmill exercise test |
| 9 | `exang` | Binary | `1`: Yes; `0`: No | Exercise-induced angina pectoris |
| 10 | `oldpeak` | Continuous | $0.0 – 6.2$ mm | ST depression induced by exercise relative to resting baseline |
| 11 | `slope` | Categorical | `1`: Upsloping<br>`2`: Flat<br>`3`: Downsloping | Slope of peak exercise ST segment |
| 12 | `ca` | Discrete | $0, 1, 2, 3$ | Number of major coronary vessels ($0–3$) colored by fluoroscopy |
| 13 | `thal` | Categorical | `3`: Normal<br>`6`: Fixed defect<br>`7`: Reversible defect | Thallium-201 myocardial scintigraphy perfusion status |
| 14 | `num` (target)| Discrete | $0$ (No), $1, 2, 3, 4$ | Angiographic coronary artery stenosis ($\ge 50\%$ diameter) |

---

### 2.4 Statlog Heart Disease Dataset (Dataset ID: 145)
- **Official Source**: UC Irvine Machine Learning Repository  
  URL: [https://archive.ics.uci.edu/dataset/145/statlog+heart](https://archive.ics.uci.edu/dataset/145/statlog+heart)
- **Total Records**: 270 instances.
- **Number of Features**: 13 numeric predictors + 1 binary class.
- **Target Variable**: `1` (Absence of heart disease), `2` (Presence of heart disease).
- **Target Distribution**: 150 absence (55.56%), 120 presence (44.44%).
- **Missing Values**: 0 missing values reported.
- **Provenance & Derivation Analysis**:  
  Statlog Heart is not an independent hospital cohort. It was curated for the 1992 Statlog project by selecting complete cases directly from the Cleveland database, recoding features, and re-labeling target 0 to 1 and targets 1–4 to 2.
- **Critical HeartGuard Rule**:  
  Statlog Heart **MUST NOT** be treated as a true external validation set for Cleveland-trained models because it shares patient records with Cleveland. In Week 2, HeartGuard AI will perform exact row matching to identify duplicate records and document the exact overlap.

---

## 3. Pipeline B: Prognostic Dataset (Framingham Heart Study Subset)

### 3.1 Dataset Profile & Provenance
- **Official Source**: National Heart, Lung, and Blood Institute (NHLBI) & Boston University  
  Official Study Portal: [https://www.framinghamheartstudy.org/](https://www.framinghamheartstudy.org/)
- **Derived Educational Dataset File**: `framingham.csv` (Circulated via Bio-Linux, Kaggle, and Biostatistics educational courses).
- **Total Records**: 4,240 individual participant records.
- **Target Variable**: `TenYearCHD` (Binary: `1` = Developed CHD within 10 years, `0` = Remained free of CHD).
- **Target Distribution**: 644 positive events (15.19%), 3,596 negative events (84.81%) — severe class imbalance.
- **Study Design**: Prospective longitudinal cohort study. Participants were free of coronary heart disease at examination baseline and followed prospectively for 10 years.

### 3.2 Provenance Verification Notice
The 4,240-row dataset corresponds to a standardized teaching subset derived from Exam 1 of the Framingham Offspring / Original Cohorts made available for epidemiological biostatistics coursework. It is not the full proprietary multi-generation NHLBI repository (which contains genetic, imaging, and multi-decade records). HeartGuard AI explicitly documents this provenance to prevent overstating data scope.

### 3.3 Framingham Feature Catalog

| Feature Name | Type | Range / Domain | Clinical Meaning |
| :--- | :--- | :--- | :--- |
| `male` | Binary | `1`: Male; `0`: Female | Biological sex |
| `age` | Integer | $32 – 70$ years | Age at baseline examination |
| `education` | Categorical | $1, 2, 3, 4$ | Educational attainment level (socioeconomic proxy) |
| `currentSmoker`| Binary | `1`: Yes; `0`: No | Current cigarette smoking status |
| `cigsPerDay` | Continuous | $0 – 70$ | Average cigarettes smoked per day |
| `BPMeds` | Binary | `1`: Yes; `0`: No | Currently on blood pressure lowering medications |
| `prevalentStroke`| Binary| `1`: Yes; `0`: No | History of documented stroke |
| `prevalentHyp` | Binary | `1`: Yes; `0`: No | Documented prevalent hypertension |
| `diabetes` | Binary | `1`: Yes; `0`: No | Documented diabetes mellitus |
| `totChol` | Continuous | $107 – 696$ mg/dL | Serum total cholesterol |
| `sysBP` | Continuous | $83.5 – 295.0$ mmHg | Systolic blood pressure (sitting) |
| `diaBP` | Continuous | $32.0 – 142.5$ mmHg | Diastolic blood pressure (sitting) |
| `BMI` | Continuous | $15.54 – 56.80$ | Body Mass Index ($kg/m^2$) |
| `heartRate` | Continuous | $44 – 143$ bpm | Resting ventricular heart rate |
| `glucose` | Continuous | $40 – 394$ mg/dL | Casual serum blood glucose |
| `TenYearCHD` | Binary (Target) | `0`: No Event; `1`: Event | Incident coronary event over 10-year follow-up |

### 3.4 Missing Value Summary (Framingham)
- `education`: 105 missing (~2.5%)
- `cigsPerDay`: 29 missing (~0.7%)
- `BPMeds`: 53 missing (~1.2%)
- `totChol`: 50 missing (~1.2%)
- `BMI`: 19 missing (~0.4%)
- `heartRate`: 1 missing (~0.02%)
- `glucose`: 388 missing (~9.1%)

---

## 4. Cross-Dataset Compatibility Matrix & Feature Harmonization

The following matrix documents feature availability and semantic alignment across all datasets:

| Standard Clinical Concept | UCI Cleveland | UCI Hungarian | UCI Long Beach | UCI Switzerland | Statlog | Framingham (Pipeline B) | Compatibility Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Age** | `age` (yrs) | `age` (yrs) | `age` (yrs) | `age` (yrs) | `age` | `age` (yrs) | Fully Compatible |
| **Sex** | `sex` (0/1) | `sex` (0/1) | `sex` (0/1) | `sex` (0/1) | `sex` | `male` (0/1) | Fully Compatible (recoded) |
| **Chest Pain Type** | `cp` (1-4) | `cp` (1-4) | `cp` (1-4) | `cp` (1-4) | `cp` (1-4) | *Not recorded* | Compatible across Pipeline A only |
| **Resting BP** | `trestbps` | `trestbps` | `trestbps` | `trestbps` | `trestbps` | `sysBP` / `diaBP` | Pipeline A uses single systolic; B has both |
| **Cholesterol** | `chol` (mg/dL) | `chol` (many 0s) | `chol` (many 0s) | `chol` (all 0s) | `chol` | `totChol` | Pipeline A has massive missingness in EU/VA |
| **Fasting Blood Sugar**| `fbs` (>120) | `fbs` (>120) | `fbs` (>120) | `fbs` (>120) | `fbs` | `diabetes` / `glucose` | Categorical threshold vs continuous glucose |
| **Resting ECG** | `restecg` (0-2)| `restecg` (0-2)| `restecg` (0-2)| `restecg` (0-2)| `restecg` | *Not recorded* | Compatible across Pipeline A only |
| **Max Heart Rate** | `thalach` | `thalach` | `thalach` | `thalach` | `thalach` | `heartRate` (Resting!) | **INCOMPATIBLE**: Treadmill max HR $\neq$ resting HR |
| **Exercise Angina** | `exang` (0/1) | `exang` (0/1) | `exang` (0/1) | `exang` (0/1) | `exang` | *Not recorded* | Compatible across Pipeline A only |
| **ST Depression** | `oldpeak` | `oldpeak` | `oldpeak` | `oldpeak` | `oldpeak` | *Not recorded* | Compatible across Pipeline A only |
| **ST Slope** | `slope` (1-3) | `slope` (missing)| `slope` (missing)| `slope` (missing)| `slope` | *Not recorded* | Compatible between Cleveland & Statlog only |
| **Fluoroscopy Vessels**| `ca` (0-3) | `ca` (99% miss.)| `ca` (missing) | `ca` (missing) | `ca` | *Not recorded* | Cleveland & Statlog only |
| **Thallium Scan** | `thal` (3,6,7) | `thal` (90% miss)| `thal` (missing) | `thal` (missing) | `thal` | *Not recorded* | Cleveland & Statlog only |
| **Clinical Target** | $\ge 50\%$ Stenosis| $\ge 50\%$ Stenosis| $\ge 50\%$ Stenosis| $\ge 50\%$ Stenosis| $\ge 50\%$ CAD | **10-Year Incident CHD** | **STRICTLY INCOMPATIBLE TARGETS** |

---

## 5. Execution Rules for Week 2 Onward

1. **Zero-Overlap External Validation**: Models trained on UCI Cleveland must be evaluated on Hungarian, Long Beach, and Switzerland only after dropping or imputing non-shared features (`ca`, `thal`, `slope`).
2. **Duplicate Detection Protocol**: Before evaluating Statlog, row-level hashing will verify exact duplicate records with Cleveland.
3. **Imputation Leakage Prevention**: Missing values in cholesterol (values $= 0$ in Hungarian and VA) must be explicitly flagged and imputed strictly inside training folds.
4. **Isolated Schemas**: Pydantic data schemas will validate input records independently for `UCIDiagnosticInput` and `FraminghamRiskInput`.
