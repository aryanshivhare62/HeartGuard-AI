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

### 2.2 Detailed Sub-Center Characteristics & Disaggregated Profiles

The four geographic databases within the UCI Heart Disease collection represent distinct clinical cohorts collected under the multicenter study led by Detrano et al. (1989 [REF-03]). They must be documented separately:

#### 1. Cleveland Clinic Foundation (Ohio, USA) — Primary Derivation Cohort
- **File**: `processed.cleveland.data`
- **Total Records ($N$)**: 303 patient cases.
- **Target Definition**: Angiographic coronary artery stenosis $\ge 50\%$ in at least one major vessel (`num` $\ge 1$).
- **Class Distribution**: 139 positive (45.87%), 164 negative (54.13%).
- **Demographics**: Mean age $54.4 \pm 9.0$ years; 206 males (68.0%), 97 females (32.0%).
- **Clinical Setting**: Patients referred for diagnostic coronary arteriography at the Cleveland Clinic Foundation.
- **Missingness Profile**: 6 missing entries across two features: `ca` (4 missing, 1.3%), `thal` (2 missing, 0.7%). All other 11 features have 0 missing values.
- **ECG Measurement Modality**: Tabular discrete parameters (`restecg`: categorical 0/1/2 based on Estes' ECG voltage and ST criteria; `oldpeak`: numeric continuous ST depression in mm; `slope`: peak exercise ST slope). **No raw ECG waveform signals or time-series tracings exist in this tabular dataset.**
- **HeartGuard Role**: Primary derivation cohort for model architecture selection, hyperparameter tuning, imputer fitting, scaler estimation, and decision threshold determination.

#### 2. Hungarian Institute of Cardiology (Budapest) — External Test Set 1
- **File**: `processed.hungarian.data`
- **Total Records ($N$)**: 294 patient cases.
- **Target Definition**: Angiographic stenosis $\ge 50\%$ (`num` $\ge 1$).
- **Class Distribution**: 106 positive (36.05%), 188 negative (63.95%) — notably lower disease prevalence than Cleveland.
- **Demographics**: Mean age $47.9 \pm 7.8$ years; 213 males (72.4%), 81 females (27.6%).
- **Clinical Setting**: Patients evaluated at the Hungarian Institute of Cardiology in Budapest.
- **Missingness Profile**:
  - `trestbps`: 1 missing (0.3%).
  - `chol`: 23 missing entries (recorded as 0 or missing).
  - `fbs`: 8 missing (2.7%).
  - `restecg`: 1 missing (0.3%).
  - `slope`: 190 missing (64.6%).
  - `ca`: 291 missing (99.0%).
  - `thal`: 266 missing (90.5%).
  *(Note: The archival documentation does not record the specific administrative or clinical reasons why fluoroscopy and thallium scintigraphy were not systematically performed or retained at this site.)*
- **HeartGuard Role**: External zero-shot validation under frozen Cleveland parameters.

#### 3. University Hospital Zurich & Basel (Switzerland) — External Test Set 2
- **File**: `processed.switzerland.data`
- **Total Records ($N$)**: 123 patient cases.
- **Target Definition**: Angiographic stenosis $\ge 50\%$ (`num` $\ge 1$).
- **Class Distribution**: 115 positive (93.50%), 8 negative (6.50%) — extreme positive class skew (spectrum bias).
- **Demographics**: Mean age $55.3 \pm 9.1$ years; 113 males (91.9%), 10 females (8.1%) — heavily male-skewed.
- **Clinical Setting**: Patients evaluated at University Hospital Zurich and Basel.
- **Missingness Profile**: `chol` is recorded as 0 across all 123 cases. The underlying reason why cholesterol is recorded as 0 in this dataset is not documented in the archival records and remains unverified; extensive missingness in exercise stress parameters (`ca`, `thal`, `slope`, `fbs`).
  *(Note: The exact administrative or clinical reasons for missing attributes in this dataset are unrecorded in the archival documentation.)*
- **HeartGuard Role**: Extreme distribution shift stress test. Evaluates model robustness when clinical prevalence is heavily skewed and exercise tests are missing.

#### 4. V.A. Medical Center (Long Beach, California) — External Test Set 3
- **File**: `processed.va.data`
- **Total Records ($N$)**: 200 patient cases.
- **Target Definition**: Angiographic stenosis $\ge 50\%$ (`num` $\ge 1$).
- **Class Distribution**: 151 positive (75.50%), 49 negative (24.50%).
- **Demographics**: Mean age $59.4 \pm 7.9$ years; 194 males (97.0%), 6 females (3.0%).
- **Clinical Setting**: Patients evaluated at the Veterans Administration Medical Center, Long Beach, California.
- **Missingness Profile**: Substantial missingness in `chol` (many 0s), `ca`, `thal`, and `slope`.
  *(Note: The exact clinical or institutional reasons for missingness are unrecorded in the archival documentation.)*
- **HeartGuard Role**: External zero-shot validation under male demographic skew, older age distribution, and high disease prevalence.

---

### 2.3 Feature Set Configurations & Statistical Justification

To prevent arbitrary data manipulation while respecting clinical reality, HeartGuard AI establishes three distinct, pre-specified feature configurations. **Crucially, each configuration is trained and tuned as a completely separate, independent model pipeline on the Cleveland derivation cohort before being frozen for external evaluation**:

1. **Full Diagnostic Set (13 Predictors)**:
   - *Features Included*: `age`, `sex`, `cp`, `trestbps`, `chol`, `fbs`, `restecg`, `thalach`, `exang`, `oldpeak`, `slope`, `ca`, `thal`.
   - *Model Derivation*: Trained exclusively on Cleveland using all 13 predictors; preprocessing, hyperparameters, and calibrators are fit on this 13-dimensional feature space.
   - *Evaluation Scope*: Restricted to Cleveland internal repeated cross-validation where fluoroscopy (`ca`) and thallium scintigraphy (`thal`) were clinically recorded for $>98\%$ of patients.
   - *Rationale*: Provides the maximal-information diagnostic benchmark.

2. **Common Reduced Diagnostic Set — 10-Feature Variant (With Cholesterol)**:
   - *Features Included*: `age`, `sex`, `cp`, `trestbps`, `chol`, `fbs`, `restecg`, `thalach`, `exang`, `oldpeak`.
   - *Features Omitted*: `slope`, `ca`, `thal` (omitted because `ca` and `thal` are missing in $90\%–99\%$ of European records; imputing them would manufacture unmeasured diagnostic tests).
   - *Model Derivation*: **Trained independently on Cleveland using strictly these 10 features**. Hyperparameters, scaling parameters, and calibrators are optimized specifically for this 10-feature model and frozen.
   - *Evaluation Scope*: Evaluated zero-shot on Hungarian ($N=294$) and Long Beach VA ($N=200$) cohorts, with missing cholesterol entries in those cohorts imputed using the frozen Cleveland training median.

3. **Common Reduced Diagnostic Set — 9-Feature Variant (Without Cholesterol)**:
   - *Features Included*: `age`, `sex`, `cp`, `trestbps`, `fbs`, `restecg`, `thalach`, `exang`, `oldpeak`.
   - *Features Omitted*: `slope`, `ca`, `thal`, **and `chol`**.
   - *Model Derivation*: **Trained independently on Cleveland using strictly these 9 features**. Hyperparameters, scaling parameters, and calibrators are optimized specifically for this 9-feature model and frozen.
   - *Evaluation Scope*: Evaluated zero-shot across **all external cohorts, specifically including Switzerland ($N=123$)**, where serum cholesterol is 100% missing (recorded as 0 across all records).
   - *Rationale*: Allows direct cross-center comparison without forcing synthetic laboratory value imputation onto an entire cohort where cholesterol was never recorded. Models do NOT drop features at test time from a 13-feature model; each configuration represents a distinct model trained on its native feature subspace.

---

### 2.4 The 14 Standard UCI Features and Canonical Schema

| Field Name | Variable Name | Data Type | Range / Domain | Clinical Description & Diagnostic Semantics |
| :--- | :--- | :--- | :--- | :--- |
| 1 | `age` | Integer | $29 – 77$ | Patient chronological age in years |
| 2 | `sex` | Binary | `1`: Male; `0`: Female | Biological sex |
| 3 | `cp` | Categorical | `1`: Typical angina<br>`2`: Atypical angina<br>`3`: Non-anginal pain<br>`4`: Asymptomatic | Chest pain type according to clinical presentation |
| 4 | `trestbps` | Continuous | $94 – 200$ mmHg | Resting systolic blood pressure on admission to hospital |
| 5 | `chol` | Continuous | $126 – 564$ mg/dL | Serum cholesterol in mg/dL (Note: values of 0 in Hungarian/VA/Swiss indicate unmeasured/missing) |
| 6 | `fbs` | Binary | `1`: $>120$ mg/dL (Elevated);<br>`0`: $\le 120$ mg/dL (Normal) | Fasting blood sugar thresholded status (categorical indicator, NOT continuous glucose) |
| 7 | `restecg` | Categorical | `0`: Normal<br>`1`: ST-T wave abnormality (T wave inversions and/or ST elevation/depression > 0.05 mV)<br>`2`: Showing probable or definite left ventricular hypertrophy by Estes' criteria | Resting electrocardiographic discrete classification |
| 8 | `thalach` | Continuous | $71 – 202$ bpm | Maximum heart rate achieved during treadmill exercise test |
| 9 | `exang` | Binary | `1`: Yes; `0`: No | Exercise-induced angina pectoris |
| 10 | `oldpeak` | Continuous | $0.0 – 6.2$ mm | ST depression induced by exercise relative to resting baseline |
| 11 | `slope` | Categorical | `1`: Upsloping<br>`2`: Flat<br>`3`: Downsloping | Slope of peak exercise ST segment |
| 12 | `ca` | Discrete | $0, 1, 2, 3$ | Number of major coronary vessels ($0–3$) colored by fluoroscopy |
| 13 | `thal` | Categorical | `3`: Normal<br>`6`: Fixed defect<br>`7`: Reversible defect | Thallium-201 myocardial scintigraphy perfusion status |
| 14 | `num` (target)| Discrete | $0$ (No), $1, 2, 3, 4$ | Angiographic coronary artery stenosis ($\ge 50\%$ diameter narrowing in $\ge 1$ major vessel) |

---

### 2.4 Statlog Heart Disease Dataset (Dataset ID: 145)
- **Official Source**: UC Irvine Machine Learning Repository  
  URL: [https://archive.ics.uci.edu/dataset/145/statlog+heart](https://archive.ics.uci.edu/dataset/145/statlog+heart)
- **Donor**: European StatLog Project (1992)
- **Total Records**: 270 instances.
- **Number of Features**: 13 numeric predictors + 1 binary class.
- **Target Variable**: `1` (Absence of heart disease), `2` (Presence of heart disease).
- **Target Distribution**: 150 absence (55.56%), 120 presence (44.44%).
- **Missing Values**: 0 missing values reported.
- **Official UCI Description & Provenance Status**:  
  According to the official UCI documentation, this dataset is a modified variant of the Cleveland database, where attributes were extracted from the larger clinical set and mapped to numeric formats.
- **Scientific Audit Correction (Verification Status)**:  
  While literature frequently treats Statlog as a subset of Cleveland complete cases, **it is methodologically incorrect to assert that all 270 records are exact duplicates of Cleveland cases prior to programmatic verification**. In Week 2, HeartGuard AI will execute exact row-level hashing and value-matching between `processed.cleveland.data` and `statlog.dat` to determine the precise overlap, recoding discrepancies, and whether any records represent distinct patients or modified entries. Until this verification is completed, Statlog is classified as a *derived benchmark under verification*, not an independent external validation cohort.

---

## 3. Pipeline B: Prognostic Dataset (Framingham Educational Subset)

### 3.1 Dataset Profile & Provenance Differentiation
- **Original Study**: The Framingham Heart Study (FHS) is a long-term, multi-generational prospective epidemiological study initiated in 1948 in Framingham, Massachusetts, under the auspices of the National Heart Institute (now NHLBI) and Boston University. The complete original repository contains multi-decade longitudinal follow-ups, genetic data, extensive lab panels, and imaging.
- **Derived Educational Dataset File**: `framingham.csv`
- **Total Records**: 4,240 participant observations.
- **Target Variable**: `TenYearCHD` (Binary: `1` = Developed incident CHD within 10-year follow-up, `0` = Remained event-free).
- **Target Distribution**: 644 positive events (15.19%), 3,596 event-free (84.81%) — class imbalance ratio ~1:5.6.
- **Provenance & Lineage Status**: **UNVERIFIED PENDING CONFIRMATION**
  The widely circulated 4,240-row dataset is commonly cited in online machine learning courseware as an educational teaching subset derived from baseline examinations of the Framingham cohort. However, the exact institutional extraction chain, data cleaning script, specific exam cohort slice, and official data transfer agreement remain unverified. It must not be cited as the official NHLBI research repository.
- **Usage Rights & Academic Licensing Status**: **UNVERIFIED PENDING CONFIRMATION**
  While widely mirrored on public platforms (e.g., Kaggle, GitHub) for educational demonstrations, formal commercial and academic redistribution licenses have not been independently confirmed with NHLBI or BioLINCC. HeartGuard AI restricts its use strictly to an offline educational machine learning baseline demonstration, pending formal confirmation of licensing terms.
- **Critical Distinction from Clinical FHS Risk Functions**:
  Official published Framingham risk scores (e.g., Wilson et al., 1998 [REF-04]; D'Agostino et al., 2008) use continuous parametric Cox proportional hazards models with specific age-stratified log-transformed variables. Machine learning models trained on this 4,240-row cross-sectional tabular file predict the empirical binary indicator `TenYearCHD` and represent an educational demonstration, not the official NHLBI clinical risk calculators.

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

### Summary of Pre-Specified Feature Configurations for Pipeline A

| Feature Configuration Name | Predictor Count | Features Included | Primary Target Cohorts | Clinical & Statistical Rationale |
| :--- | :--- | :--- | :--- | :--- |
| **Full Diagnostic Set** | 13 | `age`, `sex`, `cp`, `trestbps`, `chol`, `fbs`, `restecg`, `thalach`, `exang`, `oldpeak`, `slope`, `ca`, `thal` | Cleveland Clinic (Internal CV) | Full-information clinical diagnostic model incorporating invasive fluoroscopy and scintigraphy. |
| **Common Reduced Diagnostic Set (10-Feature Variant)** | 10 | `age`, `sex`, `cp`, `trestbps`, `chol`, `fbs`, `restecg`, `thalach`, `exang`, `oldpeak` | Cleveland, Hungarian, Long Beach VA | Drops `slope`, `ca`, `thal` to avoid manufacturing unmeasured tests. Retains cholesterol, with Hungarian/VA missing values imputed via Cleveland training statistics. |
| **Common Reduced Diagnostic Set (9-Feature Variant)** | 9 | `age`, `sex`, `cp`, `trestbps`, `fbs`, `restecg`, `thalach`, `exang`, `oldpeak` | All cohorts, specifically including Switzerland | Drops `slope`, `ca`, `thal`, **and `chol`**. Required for Switzerland where cholesterol is 100% missing, avoiding artificial imputation of lab assays. |

---

## 5. Execution Rules for Week 2 Onward

1. **Strict Cleveland-Only Model Derivation & Frozen Evaluation**:
   - Model development, feature transformation, hyperparameter search, probability calibration fitting (Platt/Isotonic), and decision threshold tuning are conducted **strictly on the Cleveland dataset** (or training folds thereof).
   - Once finalized, the entire pipeline (imputation values, scaling coefficients, model weights, calibrator parameters, decision cutoff) is **frozen**.
   - External datasets (Hungary, Long Beach VA, Switzerland) are used exclusively for zero-shot testing. **External test datasets must never be touched for tuning, feature selection, calibration fitting, or model selection.**
2. **Statlog Verification Status**:
   - Statlog will be subjected to row-hash matching against Cleveland in Week 2. Do not assume 100% duplicate identity without programmatic code test evidence.
3. **Imputation Leakage Prevention**:
   - In Hungarian and VA datasets, missing cholesterol (entries recorded as 0) must be treated as missing values, not real hypocholesterolemia. When testing the frozen Cleveland model, test records must be transformed using the median learned from the Cleveland training fold.
4. **ECG Scope Boundary**:
   - **No ECG signal processing, waveform uploads, raw voltage time-series, or deep 1D/2D CNN architectures are in scope for HeartGuard AI.**
   - All ECG information is confined to the standard discrete tabular variables (`restecg`, `oldpeak`, `slope`) recorded as clinical scalar values.
   - Any proposal to expand into ECG waveform processing is deferred and will be revisited only after Week 2 data exploration.
5. **Isolated Pydantic Schemas**:
   - Pydantic data schemas will validate input records independently for `UCIDiagnosticInput` (Pipeline A) and `FraminghamRiskInput` (Pipeline B), strictly prohibiting cross-contamination.

