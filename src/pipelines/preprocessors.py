# -*- coding: utf-8 -*-
"""
preprocessors.py
Reusable scikit-learn preprocessing pipelines for HeartGuard AI Pipeline A.

Three separate feature configurations are provided:
  - get_full_diagnostic_pipeline()      : 13-feature full diagnostic set
  - get_common_reduced_10_pipeline()    : 10-feature set (with cholesterol)
  - get_common_reduced_9_pipeline()     :  9-feature set (without cholesterol)

All transformations MUST be fitted only on training data (fit_transform on train,
transform only on test/validation). Pipelines are returned unfitted.

Pipeline B (Framingham prognostic) is strictly isolated in
get_framingham_prognostic_pipeline().
"""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ------------------------------------------------------------------
# Feature definitions
# ------------------------------------------------------------------

# Full 13-feature diagnostic set (Pipeline A primary)
FULL_NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak"]
FULL_CATEGORICAL = ["cp", "restecg", "slope", "ca", "thal"]
FULL_BINARY = ["sex", "fbs", "exang"]
FULL_FEATURES = FULL_NUMERIC + FULL_CATEGORICAL + FULL_BINARY  # 13 features

# Common Reduced 10-feature set (with cholesterol)
CR10_NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CR10_CATEGORICAL = ["cp", "restecg"]
CR10_BINARY = ["sex", "fbs", "exang"]
CR10_FEATURES = CR10_NUMERIC + CR10_CATEGORICAL + CR10_BINARY  # 10 features

# Common Reduced 9-feature set (without cholesterol)
CR9_NUMERIC = ["age", "trestbps", "thalach", "oldpeak"]
CR9_CATEGORICAL = ["cp", "restecg"]
CR9_BINARY = ["sex", "fbs", "exang"]
CR9_FEATURES = CR9_NUMERIC + CR9_CATEGORICAL + CR9_BINARY  # 9 features

# Framingham prognostic features (Pipeline B - STRICTLY ISOLATED)
FRAM_NUMERIC = ["age", "cigsPerDay", "totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose"]
FRAM_CATEGORICAL = ["education"]
FRAM_BINARY = ["male", "currentSmoker", "BPMeds", "prevalentStroke", "prevalentHyp", "diabetes"]
FRAM_FEATURES = FRAM_NUMERIC + FRAM_CATEGORICAL + FRAM_BINARY


# ------------------------------------------------------------------
# Transformer builders
# ------------------------------------------------------------------

def _numeric_transformer() -> Pipeline:
    """Median imputation + standard scaling."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])


def _categorical_transformer() -> Pipeline:
    """Most-frequent imputation + one-hot encoding (drop first level)."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)),
    ])


def _binary_transformer() -> Pipeline:
    """Most-frequent imputation, no encoding (already 0/1)."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
    ])


# ------------------------------------------------------------------
# Pipeline factories
# ------------------------------------------------------------------

def get_full_diagnostic_pipeline() -> ColumnTransformer:
    """
    13-feature full diagnostic preprocessing pipeline.
    Features: age, trestbps, chol, thalach, oldpeak (numeric)
              cp, restecg, slope, ca, thal (categorical)
              sex, fbs, exang (binary)
    """
    return ColumnTransformer(
        transformers=[
            ("numeric",      _numeric_transformer(),      FULL_NUMERIC),
            ("categorical",  _categorical_transformer(),  FULL_CATEGORICAL),
            ("binary",       _binary_transformer(),       FULL_BINARY),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def get_common_reduced_10_pipeline() -> ColumnTransformer:
    """
    10-feature Common Reduced Diagnostic Set (with cholesterol).
    Features: age, trestbps, chol, thalach, oldpeak (numeric)
              cp, restecg (categorical)
              sex, fbs, exang (binary)
    """
    return ColumnTransformer(
        transformers=[
            ("numeric",      _numeric_transformer(),      CR10_NUMERIC),
            ("categorical",  _categorical_transformer(),  CR10_CATEGORICAL),
            ("binary",       _binary_transformer(),       CR10_BINARY),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def get_common_reduced_9_pipeline() -> ColumnTransformer:
    """
    9-feature Common Reduced Diagnostic Set (without cholesterol).
    Features: age, trestbps, thalach, oldpeak (numeric)
              cp, restecg (categorical)
              sex, fbs, exang (binary)
    """
    return ColumnTransformer(
        transformers=[
            ("numeric",      _numeric_transformer(),      CR9_NUMERIC),
            ("categorical",  _categorical_transformer(),  CR9_CATEGORICAL),
            ("binary",       _binary_transformer(),       CR9_BINARY),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def get_framingham_prognostic_pipeline() -> ColumnTransformer:
    """
    Pipeline B: Framingham prognostic pipeline (STRICTLY ISOLATED from Pipeline A).
    Features: age, cigsPerDay, totChol, sysBP, diaBP, BMI, heartRate, glucose (numeric)
              education (categorical/ordinal)
              male, currentSmoker, BPMeds, prevalentStroke, prevalentHyp, diabetes (binary)
    """
    return ColumnTransformer(
        transformers=[
            ("numeric",      _numeric_transformer(),      FRAM_NUMERIC),
            ("categorical",  _categorical_transformer(),  FRAM_CATEGORICAL),
            ("binary",       _binary_transformer(),       FRAM_BINARY),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


# ------------------------------------------------------------------
# Feature name accessors (before fitting)
# ------------------------------------------------------------------

FEATURE_CONFIGS = {
    "full_13": FULL_FEATURES,
    "common_reduced_10": CR10_FEATURES,
    "common_reduced_9": CR9_FEATURES,
    "framingham_prognostic": FRAM_FEATURES,
}


def get_feature_config(name: str) -> list:
    """Return the list of raw input feature names for a named configuration."""
    if name not in FEATURE_CONFIGS:
        raise ValueError(f"Unknown config '{name}'. Choose from: {list(FEATURE_CONFIGS)}")
    return FEATURE_CONFIGS[name]

