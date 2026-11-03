# Preprocessing Documentation

> HeartGuard AI — Week 2 | `src/pipelines/preprocessors.py`

## Overview

All preprocessing transformations are implemented as scikit-learn `ColumnTransformer` pipelines. **Transformations are always fitted on training data only.** Test and validation data are transformed using the fitted statistics from training data. This is enforced and verified by anti-leakage tests in `tests/test_leakage.py`.

---

## Feature Configurations (Pipeline A — Diagnostic)

Three separate configurations are defined for UCI Heart Disease experiments. Each is returned by a dedicated factory function as an **unfitted** `ColumnTransformer`.

### Configuration 1: Full Diagnostic Set (13 features)

**Factory:** `get_full_diagnostic_pipeline()`

| Group | Features | Preprocessing |
|-------|----------|---------------|
| Numeric (5) | `age`, `trestbps`, `chol`, `thalach`, `oldpeak` | Median imputation → StandardScaler |
| Categorical (5) | `cp`, `restecg`, `slope`, `ca`, `thal` | Most-frequent imputation → OneHotEncoder (drop first) |
| Binary (3) | `sex`, `fbs`, `exang` | Most-frequent imputation → passthrough |

**Notes:**
- `ca` (0–3) and `thal` (3/6/7) have high missingness in external sites. This configuration is primarily for Cleveland-only analysis.
- `slope` is treated as categorical (values 1, 2, 3) per UCI feature description.

---

### Configuration 2: Common Reduced Set — 10 Features (with cholesterol)

**Factory:** `get_common_reduced_10_pipeline()`

| Group | Features | Preprocessing |
|-------|----------|---------------|
| Numeric (5) | `age`, `trestbps`, `chol`, `thalach`, `oldpeak` | Median imputation → StandardScaler |
| Categorical (2) | `cp`, `restecg` | Most-frequent imputation → OneHotEncoder (drop first) |
| Binary (3) | `sex`, `fbs`, `exang` | Most-frequent imputation → passthrough |

**Notes:**
- Excludes `slope`, `ca`, `thal` (high missingness in Hungary, Switzerland, VA).
- Retains `chol`; zero-coded values in Switzerland/VA are treated as valid observations unless explicitly recoded.

---

### Configuration 3: Common Reduced Set — 9 Features (without cholesterol)

**Factory:** `get_common_reduced_9_pipeline()`

| Group | Features | Preprocessing |
|-------|----------|---------------|
| Numeric (4) | `age`, `trestbps`, `thalach`, `oldpeak` | Median imputation → StandardScaler |
| Categorical (2) | `cp`, `restecg` | Most-frequent imputation → OneHotEncoder (drop first) |
| Binary (3) | `sex`, `fbs`, `exang` | Most-frequent imputation → passthrough |

**Notes:**
- Excludes `chol` in addition to `slope`, `ca`, `thal`.
- Best feature availability across all four UCI sites.
- Should be trained and evaluated separately from the 10-feature configuration.

---

## Pipeline B — Framingham Prognostic (STRICTLY ISOLATED)

**Factory:** `get_framingham_prognostic_pipeline()`

| Group | Features | Preprocessing |
|-------|----------|---------------|
| Numeric (8) | `age`, `cigsPerDay`, `totChol`, `sysBP`, `diaBP`, `BMI`, `heartRate`, `glucose` | Median imputation → StandardScaler |
| Categorical (1) | `education` | Most-frequent imputation → OneHotEncoder (drop first) |
| Binary (6) | `male`, `currentSmoker`, `BPMeds`, `prevalentStroke`, `prevalentHyp`, `diabetes` | Most-frequent imputation → passthrough |

**Isolation rules:**
- Pipeline B objects must never be fitted on UCI data.
- Pipeline B objects must never share transformer state with Pipeline A objects.
- Framingham data must not be used in any Pipeline A experiment.
- Results from Pipeline A and Pipeline B must be reported separately.

---

## Preprocessing Design Decisions

### Why median imputation for numeric features?
Median is robust to outliers and appropriate for the right-skewed distributions observed in `chol`, `trestbps`, and `oldpeak`. Mean imputation would be pulled by extreme values.

### Why most-frequent imputation for binary features?
Binary features (`sex`, `fbs`, `exang`) have very low missingness in Cleveland. Most-frequent imputation preserves the majority class and avoids introducing fractional values.

### Why OneHotEncoder with drop='first'?
Dropping the first level avoids perfect multicollinearity in linear models. `handle_unknown='ignore'` ensures external validation sites with unseen category values (e.g., `thal` coding differences) do not cause errors.

### Why not SMOTE?
SMOTE is NOT applied after one-hot encoding. Generating synthetic samples in one-hot-encoded space can produce invalid category combinations. Class imbalance will be addressed via `class_weight='balanced'` in model training (Week 3+). If oversampling is required, SMOTENC (which handles mixed feature types) will be applied before encoding on the original feature space.

### Why not use SMOTENC in Week 2?
Preprocessing pipelines are designed and tested in isolation this week. Sampling strategies will be integrated with cross-validation loops during Week 3 model training.

---

## Anti-Leakage Guarantee

Seven automated tests in `tests/test_leakage.py` verify:

| Test | What it checks |
|------|---------------|
| `test_imputer_fitted_on_train_only` | Imputer medians == training-fold medians |
| `test_scaler_fitted_on_train_only` | Scaler means == imputed training-fold means |
| `test_transform_does_not_mutate_state` | transform() on test does not change fitted state |
| `test_kfold_no_leakage` | Each CV fold uses only that fold's training rows for statistics |
| `test_pipeline_configs_are_independent` | 13F, 10F, 9F pipelines are independent objects |
| `test_framingham_pipeline_isolated` | Pipeline B shares no transformer objects with Pipeline A |
| `test_no_fit_transform_on_test` | fit_transform on test produces different output than transform |

All 7 tests pass on Python 3.13.3 with scikit-learn 1.9.1.

---

## Usage Example

```python
from src.pipelines.preprocessors import get_full_diagnostic_pipeline
from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv("data/raw/uci/cleveland.data", header=None,
                 names=["age","sex","cp","trestbps","chol","fbs",
                        "restecg","thalach","exang","oldpeak","slope","ca","thal","num"],
                 na_values="?")

FEATURES = ["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal"]
X = df[FEATURES]
y = (df["num"] >= 1).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

pipe = get_full_diagnostic_pipeline()
X_train_t = pipe.fit_transform(X_train)  # fit ONLY on training data
X_test_t  = pipe.transform(X_test)       # transform using training statistics
```

