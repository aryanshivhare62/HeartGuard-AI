# -*- coding: utf-8 -*-
"""
test_training_pipeline.py
Tests for Week 3 training pipeline correctness, leakage prevention,
and feature configuration isolation.
Run with: pytest tests/test_training_pipeline.py -v
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR / "src"))

from pipelines.preprocessors import (
    get_full_diagnostic_pipeline,
    get_common_reduced_10_pipeline,
    get_common_reduced_9_pipeline,
    FULL_FEATURES, CR10_FEATURES, CR9_FEATURES,
)
from models.model_configs import (
    get_estimator, get_param_grid, all_model_keys, MODEL_REGISTRY
)

RAW_DIR = BASE_DIR / "data" / "raw"
UCI_COLS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
]


@pytest.fixture(scope="module")
def cleveland():
    df = pd.read_csv(
        RAW_DIR / "uci" / "cleveland.data",
        header=None, names=UCI_COLS, na_values="?"
    )
    df["binary_target"] = (df["num"] >= 1).astype(int)
    return df


# ---------------------------------------------------------------------------
# 1. Data integrity
# ---------------------------------------------------------------------------

def test_cleveland_rows(cleveland):
    assert len(cleveland) == 303


def test_cleveland_binary_target_valid(cleveland):
    assert set(cleveland["binary_target"].unique()).issubset({0, 1})


def test_cleveland_disease_prevalence(cleveland):
    prev = cleveland["binary_target"].mean()
    # Cleveland prevalence should be roughly 45-55%
    assert 0.44 <= prev <= 0.56, f"Unexpected prevalence: {prev:.2f}"


# ---------------------------------------------------------------------------
# 2. Feature configuration distinctness
# ---------------------------------------------------------------------------

def test_13f_has_correct_count():
    assert len(FULL_FEATURES) == 13


def test_10f_has_correct_count():
    assert len(CR10_FEATURES) == 10


def test_9f_has_correct_count():
    assert len(CR9_FEATURES) == 9


def test_10f_contains_chol():
    assert "chol" in CR10_FEATURES


def test_9f_excludes_chol():
    assert "chol" not in CR9_FEATURES


def test_13f_contains_slope_ca_thal():
    for feat in ["slope", "ca", "thal"]:
        assert feat in FULL_FEATURES, f"13F config missing feature '{feat}'"


def test_10f_excludes_slope_ca_thal():
    for feat in ["slope", "ca", "thal"]:
        assert feat not in CR10_FEATURES, f"10F config unexpectedly contains '{feat}'"


def test_configs_are_genuinely_different():
    assert set(FULL_FEATURES) != set(CR10_FEATURES)
    assert set(CR10_FEATURES) != set(CR9_FEATURES)
    assert set(FULL_FEATURES) != set(CR9_FEATURES)


# ---------------------------------------------------------------------------
# 3. Model registry
# ---------------------------------------------------------------------------

def test_all_six_models_registered():
    keys = all_model_keys()
    expected = {
        "logistic_regression", "decision_tree", "random_forest",
        "knn", "svm", "gradient_boosting"
    }
    assert set(keys) == expected, f"Model keys: {set(keys)}"


def test_estimators_are_fresh_objects():
    """Each call to get_estimator must return a new object (not shared state)."""
    e1 = get_estimator("logistic_regression")
    e2 = get_estimator("logistic_regression")
    assert e1 is not e2


def test_param_grids_non_empty():
    for key in all_model_keys():
        grid = get_param_grid(key)
        assert len(grid) > 0, f"Empty param grid for model '{key}'"


def test_class_weight_set_for_applicable_models():
    """Models that support class_weight must have it set to 'balanced'."""
    for key in ["logistic_regression", "decision_tree", "random_forest", "svm"]:
        est = get_estimator(key)
        assert est.class_weight == "balanced", (
            f"Model '{key}' missing class_weight='balanced'"
        )


def test_svm_probability_enabled():
    """SVM must have probability=True for predict_proba / ROC-AUC."""
    svm = get_estimator("svm")
    assert svm.probability is True


# ---------------------------------------------------------------------------
# 4. Pipeline construction
# ---------------------------------------------------------------------------

def test_pipeline_builds_for_each_config():
    for factory, features in [
        (get_full_diagnostic_pipeline, FULL_FEATURES),
        (get_common_reduced_10_pipeline, CR10_FEATURES),
        (get_common_reduced_9_pipeline, CR9_FEATURES),
    ]:
        prep = factory()
        # Pipeline should have 'numeric', 'categorical', 'binary' transformers
        names = [name for name, _, _ in prep.transformers]
        assert "numeric" in names
        assert "categorical" in names
        assert "binary" in names


def test_full_sklearn_pipeline_fits(cleveland):
    X = cleveland[FULL_FEATURES]
    y = cleveland["binary_target"]
    prep = get_full_diagnostic_pipeline()
    est = get_estimator("logistic_regression")
    pipe = Pipeline([("preprocessor", prep), ("classifier", est)])
    # Should not raise
    pipe.fit(X, y)
    preds = pipe.predict(X)
    assert len(preds) == len(y)
    assert set(preds).issubset({0, 1})


def test_pipeline_produces_probabilities(cleveland):
    X = cleveland[FULL_FEATURES]
    y = cleveland["binary_target"]
    prep = get_full_diagnostic_pipeline()
    est = get_estimator("logistic_regression")
    pipe = Pipeline([("preprocessor", prep), ("classifier", est)])
    pipe.fit(X, y)
    proba = pipe.predict_proba(X)
    assert proba.shape == (len(y), 2)
    # Probabilities must sum to ~1
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# 5. Leakage prevention in CV loop
# ---------------------------------------------------------------------------

def test_preprocessing_fitted_on_train_fold_only(cleveland):
    """
    In a CV fold, the imputer medians must equal the training fold medians,
    not the validation fold medians or the full dataset medians.
    """
    X = cleveland[FULL_FEATURES]
    y = cleveland["binary_target"]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    numeric_cols = ["age", "trestbps", "chol", "thalach", "oldpeak"]

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train = X.iloc[train_idx]
        X_val = X.iloc[val_idx]

        prep = get_full_diagnostic_pipeline()
        est = get_estimator("logistic_regression")
        pipe = Pipeline([("preprocessor", prep), ("classifier", est)])
        pipe.fit(X_train, y.iloc[train_idx])

        # Imputer statistics must match training fold
        fitted_medians = pipe.named_steps["preprocessor"].named_transformers_["numeric"]["imputer"].statistics_
        expected_medians = X_train[numeric_cols].median().values

        np.testing.assert_array_almost_equal(
            fitted_medians, expected_medians, decimal=4,
            err_msg=f"Fold {fold_idx}: imputer used non-training statistics."
        )

        # Transform val fold; check pipeline state unchanged
        _ = pipe.predict(X_val)
        after_medians = pipe.named_steps["preprocessor"].named_transformers_["numeric"]["imputer"].statistics_
        np.testing.assert_array_equal(fitted_medians, after_medians,
            err_msg="Pipeline state changed after predicting on val fold.")
        break  # one fold is sufficient for this check


# ---------------------------------------------------------------------------
# 6. Config isolation — training one config must not affect another
# ---------------------------------------------------------------------------

def test_config_pipelines_are_independent_objects():
    prep_13 = get_full_diagnostic_pipeline()
    prep_10 = get_common_reduced_10_pipeline()
    prep_9 = get_common_reduced_9_pipeline()

    assert prep_13 is not prep_10
    assert prep_10 is not prep_9
    assert prep_13 is not prep_9

    # Transformers inside each must be different objects
    trans_13 = prep_13.transformers[0][1]
    trans_10 = prep_10.transformers[0][1]
    assert trans_13 is not trans_10


def test_9f_model_uses_9_features_not_13(cleveland):
    """Fitting a 9F pipeline on 9F data must not access 13F-only columns."""
    X_9 = cleveland[CR9_FEATURES]
    y = cleveland["binary_target"]
    prep = get_common_reduced_9_pipeline()
    est = get_estimator("logistic_regression")
    pipe = Pipeline([("preprocessor", prep), ("classifier", est)])
    pipe.fit(X_9, y)  # must succeed without chol/slope/ca/thal

    # 13F-only features must NOT be in input
    for feat in ["chol", "slope", "ca", "thal"]:
        assert feat not in X_9.columns, f"9F input unexpectedly contains '{feat}'"

