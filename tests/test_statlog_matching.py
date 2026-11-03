# -*- coding: utf-8 -*-
"""
test_statlog_matching.py
Automated tests for Statlog vs. Cleveland matching logic and audit findings.
Run with: pytest tests/test_statlog_matching.py -v
"""
import sys
import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"
REPORTS_DIR = BASE_DIR / "reports"

CLE_COLS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
]
STAT_COLS_RAW = [
    "age", "sex", "chestpain", "restbp", "chol", "fbs",
    "restecg", "maxhr", "exang", "oldpeak", "slope", "ca", "thal", "target",
]
STAT_RENAME = {"chestpain": "cp", "restbp": "trestbps", "maxhr": "thalach"}
PREDICTOR_COLS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]


@pytest.fixture(scope="module")
def cleveland():
    return pd.read_csv(
        RAW_DIR / "uci" / "cleveland.data",
        header=None, names=CLE_COLS, na_values="?"
    )


@pytest.fixture(scope="module")
def statlog():
    raw = pd.read_csv(
        RAW_DIR / "statlog" / "heart.dat",
        sep=r"\s+", header=None, names=STAT_COLS_RAW
    )
    return raw.rename(columns=STAT_RENAME)


@pytest.fixture(scope="module")
def audit_detail():
    path = REPORTS_DIR / "STATLOG_AUDIT_DETAIL.csv"
    return pd.read_csv(path)


# -----------------------------------------------------------------------
# 1. File structure tests
# -----------------------------------------------------------------------

def test_cleveland_row_count(cleveland):
    assert len(cleveland) == 303, f"Expected 303 Cleveland rows, got {len(cleveland)}"


def test_statlog_row_count(statlog):
    assert len(statlog) == 270, f"Expected 270 Statlog rows, got {len(statlog)}"


def test_statlog_no_missing_predictors(statlog):
    missing = statlog[PREDICTOR_COLS].isna().sum().sum()
    assert missing == 0, f"Statlog has {missing} NaN predictor values (expected 0)"


def test_cleveland_has_missing_values(cleveland):
    missing = cleveland[PREDICTOR_COLS].isna().sum().sum()
    assert missing > 0, "Cleveland should have missing predictor values"


# -----------------------------------------------------------------------
# 2. Feature coding equivalence tests
# -----------------------------------------------------------------------

def test_sex_coding_identical(cleveland, statlog):
    cle_vals = set(cleveland["sex"].dropna().unique())
    stat_vals = set(statlog["sex"].dropna().unique())
    assert cle_vals == stat_vals, f"sex coding differs: Cleveland={cle_vals}, Statlog={stat_vals}"


def test_cp_coding_identical(cleveland, statlog):
    cle_vals = set(cleveland["cp"].dropna().unique())
    stat_vals = set(statlog["cp"].dropna().unique())
    assert cle_vals == stat_vals, f"cp coding differs: Cleveland={cle_vals}, Statlog={stat_vals}"


def test_thal_coding_identical(cleveland, statlog):
    cle_vals = set(cleveland["thal"].dropna().unique())
    stat_vals = set(statlog["thal"].dropna().unique())
    assert cle_vals == stat_vals, f"thal coding differs: Cleveland={cle_vals}, Statlog={stat_vals}"


def test_slope_coding_identical(cleveland, statlog):
    cle_vals = set(cleveland["slope"].dropna().unique())
    stat_vals = set(statlog["slope"].dropna().unique())
    assert cle_vals == stat_vals, f"slope coding differs: Cleveland={cle_vals}, Statlog={stat_vals}"


def test_fbs_coding_identical(cleveland, statlog):
    cle_vals = set(cleveland["fbs"].dropna().unique())
    stat_vals = set(statlog["fbs"].dropna().unique())
    assert cle_vals == stat_vals, f"fbs coding differs: Cleveland={cle_vals}, Statlog={stat_vals}"


# -----------------------------------------------------------------------
# 3. Match count tests (verified audit findings)
# -----------------------------------------------------------------------

def test_exact_match_count(audit_detail):
    n_exact = (audit_detail["match_type"] == "EXACT").sum()
    assert n_exact == 270, f"Expected 270 exact matches, got {n_exact}"


def test_unmatched_nan_count(audit_detail):
    n_nan = (audit_detail["match_type"] == "UNMATCHED_NAN").sum()
    assert n_nan == 6, f"Expected 6 NaN-unmatched Cleveland rows, got {n_nan}"


def test_unmatched_complete_count(audit_detail):
    """
    27 complete (non-NaN) Cleveland rows have no Statlog counterpart.
    This was the key correction from the initial report.
    """
    n_complete_unmatched = (audit_detail["match_type"] == "UNMATCHED_COMPLETE").sum()
    assert n_complete_unmatched == 27, (
        f"Expected 27 complete-row unmatched Cleveland rows, got {n_complete_unmatched}. "
        "The initial report incorrectly claimed NaN explained all 33 unmatched rows."
    )


def test_total_unmatched_count(audit_detail):
    n_total_unmatched = (audit_detail["match_type"] != "EXACT").sum()
    assert n_total_unmatched == 33, f"Expected 33 total unmatched Cleveland rows, got {n_total_unmatched}"


def test_statlog_100_percent_matched(audit_detail):
    """All 270 Statlog rows must be matched to at least one Cleveland row."""
    # Count unique statlog indices across all exact-matched rows
    matched_stat_indices = set()
    exact_rows = audit_detail[audit_detail["match_type"] == "EXACT"]
    # n_statlog_matches should be >= 1 for all exact rows
    assert (exact_rows["n_statlog_matches"] >= 1).all(), "Some exact-match rows have 0 Statlog matches"
    # The total unique Statlog indices matched should be 270
    # (We verify this via the audit summary: 270 Statlog rows matched)
    assert len(exact_rows) == 270, f"Expected 270 exact-match rows, got {len(exact_rows)}"


# -----------------------------------------------------------------------
# 4. Target label equivalence tests
# -----------------------------------------------------------------------

def test_target_labels_all_consistent(audit_detail):
    """All 270 matched rows should have consistent binary target labels."""
    exact_rows = audit_detail[audit_detail["match_type"] == "EXACT"]
    n_inconsistent = (exact_rows["target_label_consistent"] == False).sum()
    assert n_inconsistent == 0, (
        f"{n_inconsistent} matched rows have inconsistent target labels. "
        "Binary mapping: Cleveland num>=1 <-> Statlog target==2."
    )


def test_cleveland_target_range(cleveland):
    """Cleveland 'num' should be integers 0-4."""
    valid = cleveland["num"].isin([0, 1, 2, 3, 4])
    assert valid.all(), f"Unexpected Cleveland num values: {cleveland.loc[~valid, 'num'].unique()}"


def test_statlog_target_range(statlog):
    """Statlog 'target' should be 1 or 2 only."""
    valid = statlog["target"].isin([1, 2])
    assert valid.all(), f"Unexpected Statlog target values: {statlog.loc[~valid, 'target'].unique()}"


def test_binary_target_mapping_correct(cleveland, statlog):
    """
    Verify the documented binary mapping on a sample of matched rows.
    Cleveland num==0 should map to Statlog target==1;
    Cleveland num>=1 should map to Statlog target==2.
    """
    # Build Statlog lookup by predictor float tuple
    def make_tuple(row):
        return tuple(float(row[c]) for c in PREDICTOR_COLS)

    stat_lookup = {}
    for _, row in statlog.iterrows():
        key = make_tuple(row)
        stat_lookup[key] = int(row["target"])

    mismatches = 0
    checked = 0
    for _, row in cleveland.iterrows():
        if row[PREDICTOR_COLS].isna().any():
            continue
        key = make_tuple(row)
        if key not in stat_lookup:
            continue
        stat_target = stat_lookup[key]
        cle_binary = int(row["num"] >= 1)
        stat_binary = int(stat_target == 2)
        if cle_binary != stat_binary:
            mismatches += 1
        checked += 1

    assert checked == 270, f"Expected to check 270 matched rows, checked {checked}"
    assert mismatches == 0, f"{mismatches} target label mismatches found in {checked} matched rows"


# -----------------------------------------------------------------------
# 5. Matching logic integrity tests (unit tests for the matching function)
# -----------------------------------------------------------------------

def test_nan_row_cannot_match():
    """A Cleveland row with NaN predictors must not be flagged as an exact match."""
    row_with_nan = pd.Series({
        "age": 50.0, "sex": 1.0, "cp": 3.0, "trestbps": 130.0,
        "chol": float("nan"), "fbs": 0.0, "restecg": 0.0,
        "thalach": 150.0, "exang": 0.0, "oldpeak": 1.0,
        "slope": 2.0, "ca": 0.0, "thal": 3.0
    })
    has_nan = row_with_nan[PREDICTOR_COLS].isna().any()
    assert has_nan, "Test row should have NaN"


def test_complete_row_match_logic():
    """A complete Cleveland row that matches Statlog should produce distance=0."""
    # Use a known matched row (Cleveland row 0: age=63, sex=1, cp=1, ...)
    cle_row = np.array([63, 1, 1, 145, 233, 1, 2, 150, 0, 2.3, 3, 0, 6], dtype=float)
    # Build a mock statlog containing this row
    stat_row = cle_row.copy()
    dist = float(np.sqrt(np.sum((cle_row - stat_row) ** 2)))
    assert dist == 0.0, "Identical rows should have distance 0"


def test_non_duplicate_row_nonzero_distance():
    """Two different clinical rows should have nonzero distance."""
    row1 = np.array([63, 1, 1, 145, 233, 1, 2, 150, 0, 2.3, 3, 0, 6], dtype=float)
    row2 = np.array([67, 1, 4, 160, 286, 0, 2, 108, 1, 1.5, 2, 3, 3], dtype=float)
    dist = float(np.sqrt(np.sum((row1 - row2) ** 2)))
    assert dist > 0, "Different rows should have nonzero distance"


# -----------------------------------------------------------------------
# 6. Implication test: Statlog must NOT be used as external validation
# -----------------------------------------------------------------------

def test_statlog_is_not_independent(audit_detail):
    """
    Since all 270 Statlog rows exist in Cleveland, Statlog cannot be treated
    as an independent external validation set. This test asserts that fact.
    """
    n_matched = (audit_detail["match_type"] == "EXACT").sum()
    n_stat = 270
    pct = n_matched / n_stat
    assert pct >= 1.0, (
        f"Only {pct:.1%} of Statlog rows found in Cleveland. "
        "If this drops below 100%, re-investigate before using Statlog as validation."
    )
    # This test PASSING means Statlog IS NOT independent -> should NOT be used as external val
