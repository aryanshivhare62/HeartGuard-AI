# -*- coding: utf-8 -*-
"""
test_leakage.py
Anti-data-leakage tests for HeartGuard AI preprocessing pipelines.

Tests verify that:
1. Imputation statistics (median, mode) are derived ONLY from training data.
2. Scaler statistics (mean, std) are derived ONLY from training data.
3. Fitting on fold K1 does not alter state when transforming fold K2.
4. Pipeline configurations are independent (full / CR10 / CR9 do not share state).
5. Pipeline B (Framingham) state is completely isolated from Pipeline A state.

Run with: pytest tests/test_leakage.py -v
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold

# Make src importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pipelines.preprocessors import (
    get_common_reduced_10_pipeline,
    get_common_reduced_9_pipeline,
    get_framingham_prognostic_pipeline,
    get_full_diagnostic_pipeline,
    CR9_FEATURES,
    CR10_FEATURES,
    FULL_FEATURES,
    FRAM_FEATURES,
)

# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"

UCI_COLS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
]


def _load_cleveland() -> pd.DataFrame:
    path = RAW_DIR / "uci" / "cleveland.data"
    df = pd.read_csv(path, header=None, names=UCI_COLS, na_values="?")
    df["binary_target"] = (df["num"] >= 1).astype(int)
    return df


@pytest.fixture(scope="module")
def cleveland():
    return _load_cleveland()


@pytest.fixture(scope="module")
def mock_split(cleveland):
    """Return a single deterministic 80/20 train/test split."""
    from sklearn.model_selection import train_test_split
    X = cleveland[FULL_FEATURES]
    y = cleveland["binary_target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test


# ------------------------------------------------------------------
# Helper: extract numeric imputer medians and scaler means
# ------------------------------------------------------------------

def _get_imputer_medians(pipeline) -> np.ndarray:
    """Extract fitted imputer statistics from a ColumnTransformer."""
    return pipeline.named_transformers_["numeric"]["imputer"].statistics_


def _get_scaler_means(pipeline) -> np.ndarray:
    """Extract fitted scaler means from a ColumnTransformer."""
    return pipeline.named_transformers_["numeric"]["scaler"].mean_


# ------------------------------------------------------------------
# Test 1: Imputation uses training-fold statistics only
# ------------------------------------------------------------------

def test_imputer_fitted_on_train_only(mock_split):
    """Imputer medians must match training data medians, not test data medians."""
    X_train, X_test, _, _ = mock_split
    pipe = get_full_diagnostic_pipeline()
    pipe.fit(X_train)

    fitted_medians = _get_imputer_medians(pipe)
    numeric_cols = ["age", "trestbps", "chol", "thalach", "oldpeak"]
    expected_medians = X_train[numeric_cols].median().values

    np.testing.assert_array_almost_equal(
        fitted_medians, expected_medians,
        decimal=5,
        err_msg="Imputer statistics do not match training data medians.",
    )

    # Verify they differ from test-set medians (as a sanity check that test != train medians)
    # (This may trivially pass if they happen to be identical; it's informational.)
    test_medians = X_test[numeric_cols].median().values
    # We just confirm the pipeline did NOT use test medians
    is_test_median = np.allclose(fitted_medians, test_medians, atol=1e-3)
    # We do not assert False here because they COULD coincidentally match;
    # the real assertion is that fitted_medians == train_medians (above).
    print(f"Train medians == Test medians: {is_test_median} (informational only)")


# ------------------------------------------------------------------
# Test 2: Scaler uses training-fold statistics only
# ------------------------------------------------------------------

def test_scaler_fitted_on_train_only(mock_split):
    """Scaler means must match training data column means."""
    X_train, X_test, _, _ = mock_split
    pipe = get_full_diagnostic_pipeline()
    # After imputation, scaler sees imputed values; so we compare on imputed train data
    pipe.fit(X_train)

    fitted_means = _get_scaler_means(pipe)

    # Impute train data manually to get the same imputed values
    imputer = pipe.named_transformers_["numeric"]["imputer"]
    numeric_cols = ["age", "trestbps", "chol", "thalach", "oldpeak"]
    X_train_imputed = imputer.transform(X_train[numeric_cols])
    expected_means = X_train_imputed.mean(axis=0)

    np.testing.assert_array_almost_equal(
        fitted_means, expected_means, decimal=5,
        err_msg="Scaler means do not match imputed training data means.",
    )


# ------------------------------------------------------------------
# Test 3: transform() on test fold does not alter fitted state
# ------------------------------------------------------------------

def test_transform_does_not_mutate_state(mock_split):
    """After fitting on train, transforming test must not change pipeline state."""
    X_train, X_test, _, _ = mock_split
    pipe = get_full_diagnostic_pipeline()
    pipe.fit(X_train)

    medians_before = _get_imputer_medians(pipe).copy()
    means_before = _get_scaler_means(pipe).copy()

    _ = pipe.transform(X_test)

    medians_after = _get_imputer_medians(pipe)
    means_after = _get_scaler_means(pipe)

    np.testing.assert_array_equal(
        medians_before, medians_after,
        err_msg="Imputer medians changed after transform() on test fold.",
    )
    np.testing.assert_array_equal(
        means_before, means_after,
        err_msg="Scaler means changed after transform() on test fold.",
    )


# ------------------------------------------------------------------
# Test 4: Cross-fold independence (K-fold leak test)
# ------------------------------------------------------------------

def test_kfold_no_leakage(cleveland):
    """
    In a K-fold loop, each fold's imputer/scaler statistics must match only
    that fold's training rows, not the held-out validation rows.
    """
    X = cleveland[FULL_FEATURES]
    y = cleveland["binary_target"]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    numeric_cols = ["age", "trestbps", "chol", "thalach", "oldpeak"]

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train = X.iloc[train_idx]
        X_val = X.iloc[val_idx]

        pipe = get_full_diagnostic_pipeline()
        pipe.fit(X_train)

        fitted_medians = _get_imputer_medians(pipe)
        expected_medians = X_train[numeric_cols].median().values

        np.testing.assert_array_almost_equal(
            fitted_medians, expected_medians, decimal=5,
            err_msg=f"Fold {fold_idx}: imputer medians do not match training fold medians.",
        )

        # Confirm val set stats were NOT used
        val_medians = X_val[numeric_cols].median().values
        # fitted_medians should equal train medians, not val medians
        # (Only flagged if they differ; they could coincidentally equal val medians)
        if not np.allclose(fitted_medians, val_medians, atol=0.5):
            assert np.allclose(fitted_medians, expected_medians, atol=1e-5), (
                f"Fold {fold_idx}: pipeline leaked validation data into imputation."
            )


# ------------------------------------------------------------------
# Test 5: Pipeline configurations are independent objects
# ------------------------------------------------------------------

def test_pipeline_configs_are_independent(mock_split):
    """Fitting one pipeline config must not affect another config's state."""
    X_train, X_test, _, _ = mock_split

    pipe_13 = get_full_diagnostic_pipeline()
    pipe_10 = get_common_reduced_10_pipeline()
    pipe_9 = get_common_reduced_9_pipeline()

    pipe_13.fit(X_train)
    pipe_10.fit(X_train)

    # Fit 13F and 10F; 9F should still be unfitted
    with pytest.raises(Exception):
        # Unfitted pipeline should raise on transform
        _ = pipe_9.transform(X_train[CR9_FEATURES])

    # Now fit 9F separately
    pipe_9.fit(X_train)

    # Each pipeline's scaler means should differ from others if features differ
    # (At minimum, 9F has fewer numeric features than 13F/10F)
    means_13 = _get_scaler_means(pipe_13)
    means_9 = _get_scaler_means(pipe_9)
    assert means_13.shape != means_9.shape or not np.allclose(means_13, means_9), (
        "Full (13F) and CR9 pipelines have identical scaler state; expected differences."
    )


# ------------------------------------------------------------------
# Test 6: Pipeline B (Framingham) is isolated from Pipeline A
# ------------------------------------------------------------------

def test_framingham_pipeline_isolated():
    """
    Pipeline B (Framingham prognostic) uses different features from Pipeline A.
    It must not share transformer objects with any Pipeline A config.
    Checks are structural (pre-fit) to avoid calling fitted-only properties.
    """
    pipe_a = get_full_diagnostic_pipeline()
    pipe_b = get_framingham_prognostic_pipeline()

    # They should be different objects
    assert pipe_a is not pipe_b, "Pipeline A and Pipeline B are the same object."

    # Access transformers via .transformers (pre-fit attribute), not .transformers_ (post-fit)
    trans_a_obj = pipe_a.transformers[0][1]   # numeric transformer for Pipeline A
    trans_b_obj = pipe_b.transformers[0][1]   # numeric transformer for Pipeline B

    assert trans_a_obj is not trans_b_obj, (
        "Pipeline A and B share the same numeric transformer object (they must be independent)."
    )

    # Structural: verify Pipeline A numeric features
    a_numeric_cols = pipe_a.transformers[0][2]
    assert a_numeric_cols == ["age", "trestbps", "chol", "thalach", "oldpeak"], (
        f"Pipeline A numeric features unexpected: {a_numeric_cols}"
    )

    # Structural: verify Pipeline B numeric features are different
    b_numeric_cols = pipe_b.transformers[0][2]
    assert b_numeric_cols == ["age", "cigsPerDay", "totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose"], (
        f"Pipeline B numeric features unexpected: {b_numeric_cols}"
    )

    # No Pipeline-A-exclusive feature should appear in Pipeline B
    pipeline_a_only = {"trestbps", "chol", "thalach", "oldpeak", "cp", "restecg",
                        "slope", "ca", "thal", "fbs", "exang"}
    for feat in b_numeric_cols + pipe_b.transformers[1][2] + pipe_b.transformers[2][2]:
        assert feat not in pipeline_a_only, (
            f"Pipeline B contains Pipeline A feature '{feat}'; pipelines are not isolated."
        )


# ------------------------------------------------------------------
# Test 7: No fit_transform on test fold (integration smoke test)
# ------------------------------------------------------------------

def test_no_fit_transform_on_test(mock_split):
    """
    Calling fit_transform on test fold would leak information.
    Verify that using transform() instead of fit_transform() on test data
    produces deterministic output consistent with training-derived statistics.
    """
    X_train, X_test, _, _ = mock_split
    pipe_a = get_full_diagnostic_pipeline()
    pipe_b = get_full_diagnostic_pipeline()

    # Correct usage: fit on train, transform on test
    X_train_transformed = pipe_a.fit_transform(X_train)
    X_test_transformed = pipe_a.transform(X_test)

    # Incorrect usage: fit_transform on test (leaks test statistics)
    X_test_leaked = pipe_b.fit_transform(X_test)

    # The two transforms of test should differ because statistics differ
    assert not np.allclose(X_test_transformed, X_test_leaked, atol=1e-3), (
        "Correct transform and leaked transform produced identical results; "
        "test data may be degenerate or train/test statistics happen to be identical."
    )
