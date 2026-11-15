# -*- coding: utf-8 -*-
"""
train_baseline_models.py
Week 3 — HeartGuard AI Pipeline A Baseline Model Training.

Trains 6 baseline classifiers across 3 feature configurations on the
Cleveland dataset using Repeated Stratified 5-Fold CV (10 repeats, 50 splits).

Strict rules enforced:
  - Only Cleveland data (N=303) used for training and internal validation.
  - Statlog, Hungary, Switzerland, VA Long Beach are NOT used here.
  - All preprocessing is fitted inside training folds only.
  - Three feature configurations are trained as genuinely separate models.
  - Class weighting is the primary imbalance strategy (no SMOTE after encoding).
  - Hyperparameter search uses inner CV to avoid leaking val-fold statistics.
  - Random seeds are fixed throughout.

Outputs:
  results/baseline/<config>/  -- per-model CSV results
  models/baseline/<config>/   -- saved fitted Pipeline objects (joblib)
  reports/BASELINE_MODEL_RESULTS.md
"""
import sys
import json
import warnings
from pathlib import Path
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    RepeatedStratifiedKFold,
    StratifiedKFold,
)
from sklearn.pipeline import Pipeline

# Project imports
BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR / "src"))

from pipelines.preprocessors import (
    get_full_diagnostic_pipeline,
    get_common_reduced_10_pipeline,
    get_common_reduced_9_pipeline,
    FULL_FEATURES,
    CR10_FEATURES,
    CR9_FEATURES,
)
from models.model_configs import MODEL_REGISTRY, all_model_keys, get_estimator

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
N_REPEATS = 10
N_SPLITS = 5          # 5-fold x 10 repeats = 50 splits
INNER_CV_SPLITS = 3   # inner CV for GridSearchCV (conservative on N~240 train)
SCORING_INNER = "roc_auc"

RAW_DIR = BASE_DIR / "data" / "raw"
RESULTS_DIR = BASE_DIR / "results" / "baseline"
MODELS_DIR = BASE_DIR / "models" / "baseline"
REPORTS_DIR = BASE_DIR / "reports"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Feature configuration registry
# ---------------------------------------------------------------------------
FEATURE_CONFIGS = {
    "full_13f": {
        "display": "Full Diagnostic Set (13 features)",
        "features": FULL_FEATURES,
        "preprocessor_factory": get_full_diagnostic_pipeline,
    },
    "reduced_10f": {
        "display": "Common Reduced Set — 10 features (with cholesterol)",
        "features": CR10_FEATURES,
        "preprocessor_factory": get_common_reduced_10_pipeline,
    },
    "reduced_9f": {
        "display": "Common Reduced Set — 9 features (without cholesterol)",
        "features": CR9_FEATURES,
        "preprocessor_factory": get_common_reduced_9_pipeline,
    },
}

# ---------------------------------------------------------------------------
# Load Cleveland data
# ---------------------------------------------------------------------------
UCI_COLS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
]

cleveland_path = RAW_DIR / "uci" / "cleveland.data"
cleveland = pd.read_csv(cleveland_path, header=None, names=UCI_COLS, na_values="?")
y_full = (cleveland["num"] >= 1).astype(int)

print(f"Cleveland loaded: {len(cleveland)} rows")
print(f"Class distribution: {y_full.value_counts().to_dict()}")
print(f"Disease prevalence: {y_full.mean():.1%}\n")

# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

def evaluate_fold(y_true, y_pred, y_proba):
    """Compute all metrics for a single fold."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "accuracy":    accuracy_score(y_true, y_pred),
        "sensitivity": recall_score(y_true, y_pred, zero_division=0),   # TPR
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else float("nan"),
        "precision":   precision_score(y_true, y_pred, zero_division=0),
        "f1":          f1_score(y_true, y_pred, zero_division=0),
        "roc_auc":     roc_auc_score(y_true, y_proba),
        "pr_auc":      average_precision_score(y_true, y_proba),
        "brier":       brier_score_loss(y_true, y_proba),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }


METRIC_COLS = [
    "accuracy", "sensitivity", "specificity", "precision",
    "f1", "roc_auc", "pr_auc", "brier",
]

# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
all_summary_rows = []   # for the markdown report

for config_key, config in FEATURE_CONFIGS.items():
    print(f"\n{'='*60}")
    print(f"CONFIGURATION: {config['display']}")
    print(f"{'='*60}")

    features = config["features"]
    X = cleveland[features]
    y = y_full

    config_results_dir = RESULTS_DIR / config_key
    config_results_dir.mkdir(parents=True, exist_ok=True)
    config_models_dir = MODELS_DIR / config_key
    config_models_dir.mkdir(parents=True, exist_ok=True)

    # Outer CV: Repeated Stratified 5-Fold (10 repeats = 50 splits)
    rskf = RepeatedStratifiedKFold(
        n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_SEED
    )

    for model_key in all_model_keys():
        model_info = MODEL_REGISTRY[model_key]
        print(f"\n  Model: {model_info['display_name']}")

        fold_results = []

        for fold_idx, (train_idx, val_idx) in enumerate(rskf.split(X, y)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # Build fresh pipeline: preprocessor + estimator
            preprocessor = config["preprocessor_factory"]()
            base_estimator = get_estimator(model_key)

            # Wrap in full sklearn Pipeline
            full_pipeline = Pipeline([
                ("preprocessor", preprocessor),
                ("classifier", base_estimator),
            ])

            # Inner CV for hyperparameter selection
            param_grid_prefixed = {
                f"classifier__{k}": v
                for k, v in MODEL_REGISTRY[model_key]["param_grid"].items()
            }

            inner_cv = StratifiedKFold(
                n_splits=INNER_CV_SPLITS, shuffle=True, random_state=RANDOM_SEED + fold_idx
            )

            gs = GridSearchCV(
                estimator=full_pipeline,
                param_grid=param_grid_prefixed,
                cv=inner_cv,
                scoring=SCORING_INNER,
                n_jobs=1,
                refit=True,
            )

            # Fit on training fold (preprocessing fitted here, never on val)
            gs.fit(X_train, y_train)

            # Predict on held-out validation fold
            best_pipeline = gs.best_estimator_
            y_pred = best_pipeline.predict(X_val)
            y_proba = best_pipeline.predict_proba(X_val)[:, 1]

            metrics = evaluate_fold(y_val, y_pred, y_proba)
            metrics["fold"] = fold_idx
            metrics["repeat"] = fold_idx // N_SPLITS
            metrics["inner_fold"] = fold_idx % N_SPLITS
            metrics["best_params"] = str(gs.best_params_)
            fold_results.append(metrics)

        # Aggregate across 50 folds
        fold_df = pd.DataFrame(fold_results)
        fold_csv = config_results_dir / f"{model_key}_fold_results.csv"
        fold_df.to_csv(fold_csv, index=False, encoding="utf-8")

        # Summary statistics
        means = fold_df[METRIC_COLS].mean()
        stds = fold_df[METRIC_COLS].std()

        print(f"    ROC-AUC: {means['roc_auc']:.3f} +/- {stds['roc_auc']:.3f}")
        print(f"    F1:      {means['f1']:.3f} +/- {stds['f1']:.3f}")
        print(f"    Brier:   {means['brier']:.3f} +/- {stds['brier']:.3f}")

        # Save final model fitted on ALL Cleveland data (for artefact preservation)
        # This is NOT used for metric reporting; metrics come from CV only.
        final_preprocessor = config["preprocessor_factory"]()
        final_estimator = get_estimator(model_key)
        final_pipeline = Pipeline([
            ("preprocessor", final_preprocessor),
            ("classifier", final_estimator),
        ])
        # Use best params from last fold as a representative (do not tune on full data)
        last_best_params = {
            k.replace("classifier__", ""): v
            for k, v in gs.best_params_.items()
        }
        final_pipeline.named_steps["classifier"].set_params(**last_best_params)
        final_pipeline.fit(X, y)

        model_path = config_models_dir / f"{model_key}.joblib"
        joblib.dump(final_pipeline, model_path)

        # Collect for report
        summary_row = {
            "config": config["display"],
            "model": model_info["display_name"],
            "model_key": model_key,
            "config_key": config_key,
        }
        for metric in METRIC_COLS:
            summary_row[f"{metric}_mean"] = round(float(means[metric]), 4)
            summary_row[f"{metric}_std"] = round(float(stds[metric]), 4)
        all_summary_rows.append(summary_row)

    print(f"\n  Config '{config_key}' complete. Results saved to {config_results_dir}")

# ---------------------------------------------------------------------------
# Write summary CSV
# ---------------------------------------------------------------------------
summary_df = pd.DataFrame(all_summary_rows)
summary_csv = RESULTS_DIR / "baseline_summary.csv"
summary_df.to_csv(summary_csv, index=False, encoding="utf-8")
print(f"\nSummary CSV written to: {summary_csv}")

# ---------------------------------------------------------------------------
# Write BASELINE_MODEL_RESULTS.md
# ---------------------------------------------------------------------------
report_path = REPORTS_DIR / "BASELINE_MODEL_RESULTS.md"
with open(report_path, "w", encoding="utf-8") as f:
    f.write("# Baseline Model Results\n\n")
    f.write("> HeartGuard AI - Week 3 | Pipeline A — Cleveland Dataset (N=303)\n")
    f.write(f"> Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")

    f.write("## Methodology\n\n")
    f.write("- **Dataset:** Cleveland (N=303) only. No external sites used.\n")
    f.write("- **Validation:** Repeated Stratified 5-Fold CV, 10 repeats (50 splits total).\n")
    f.write("- **Hyperparameter selection:** Inner 3-fold GridSearchCV within each training fold, optimising ROC-AUC.\n")
    f.write("- **Imbalance strategy:** class_weight='balanced' (or equivalent). No SMOTE.\n")
    f.write("- **Metrics reported:** Mean +/- SD across 50 folds.\n")
    f.write("- **Limitation:** 50 fold results are correlated (10 repeats share the same observations). "
            "SD reflects fold-to-fold variability, not independent replications. "
            "Bootstrap CIs on the full cohort will be computed separately.\n\n")

    for config_key, config in FEATURE_CONFIGS.items():
        f.write(f"## {config['display']}\n\n")
        config_rows = [r for r in all_summary_rows if r["config_key"] == config_key]

        # ROC-AUC table
        f.write("### ROC-AUC (mean +/- SD across 50 CV folds)\n\n")
        f.write("| Model | ROC-AUC | PR-AUC | F1 | Sensitivity | Specificity | Accuracy | Brier |\n")
        f.write("|-------|---------|--------|----|-------------|-------------|----------|-------|\n")
        for row in sorted(config_rows, key=lambda r: -r["roc_auc_mean"]):
            f.write(
                f"| {row['model']} "
                f"| {row['roc_auc_mean']:.3f} +/- {row['roc_auc_std']:.3f} "
                f"| {row['pr_auc_mean']:.3f} +/- {row['pr_auc_std']:.3f} "
                f"| {row['f1_mean']:.3f} +/- {row['f1_std']:.3f} "
                f"| {row['sensitivity_mean']:.3f} +/- {row['sensitivity_std']:.3f} "
                f"| {row['specificity_mean']:.3f} +/- {row['specificity_std']:.3f} "
                f"| {row['accuracy_mean']:.3f} +/- {row['accuracy_std']:.3f} "
                f"| {row['brier_mean']:.3f} +/- {row['brier_std']:.3f} |\n"
            )
        f.write("\n")

    f.write("## Important Caveats\n\n")
    f.write("1. All metrics are internal CV estimates on Cleveland (N=303). "
            "External generalisation is unknown until zero-shot evaluation on Hungary, "
            "Switzerland, and VA Long Beach in Week 5.\n")
    f.write("2. Hyperparameters were selected via inner CV to avoid outer-fold leakage. "
            "The inner CV uses only training-fold data for fitting and selection.\n")
    f.write("3. The 50 CV fold results are statistically correlated (not independent). "
            "SD is an underestimate of true uncertainty. "
            "Non-parametric bootstrap CIs (B=1,000) will be computed in Week 4.\n")
    f.write("4. These are baseline results with bounded hyperparameter grids. "
            "Advanced ensembles and Optuna tuning occur in Week 4.\n")
    f.write("5. Statlog was confirmed to be a subset of Cleveland and is NOT used "
            "for training or validation.\n")

print(f"\nBaseline results report written to: {report_path}")
print("\nWeek 3 training complete.")

