# -*- coding: utf-8 -*-
"""
model_configs.py
Defines the 6 baseline classifiers with bounded hyperparameter grids
appropriate for the small Cleveland dataset (N=303).

Design rules:
- All models use class_weight='balanced' or equivalent to handle imbalance.
- Hyperparameter grids are intentionally narrow to prevent overfitting on N=303.
- No SMOTE is applied; class weighting is the primary imbalance strategy.
- Random states are fixed for reproducibility.
- Each entry returns a fresh (unfitted) estimator.
"""
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Global reproducibility seed
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Model registry
# Each entry: (name, estimator_factory, param_grid)
# param_grid keys must match estimator constructor argument names.
# ---------------------------------------------------------------------------

MODEL_REGISTRY = {
    "logistic_regression": {
        "display_name": "Logistic Regression",
        "estimator": lambda: LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            solver="lbfgs",
            random_state=RANDOM_SEED,
        ),
        # C controls regularisation strength (1/lambda). Small grid for N=303.
        "param_grid": {
            "C": [0.01, 0.1, 1.0, 10.0],
            "penalty": ["l2"],
        },
    },
    "decision_tree": {
        "display_name": "Decision Tree",
        "estimator": lambda: DecisionTreeClassifier(
            class_weight="balanced",
            random_state=RANDOM_SEED,
        ),
        # Tight depth constraint avoids overfitting on small N.
        "param_grid": {
            "max_depth": [3, 4, 5, 6],
            "min_samples_leaf": [5, 10, 15],
            "criterion": ["gini", "entropy"],
        },
    },
    "random_forest": {
        "display_name": "Random Forest",
        "estimator": lambda: RandomForestClassifier(
            class_weight="balanced",
            random_state=RANDOM_SEED,
            n_jobs=1,
        ),
        "param_grid": {
            "n_estimators": [100, 200],
            "max_depth": [3, 5, None],
            "min_samples_leaf": [3, 5],
            "max_features": ["sqrt", "log2"],
        },
    },
    "knn": {
        "display_name": "K-Nearest Neighbours",
        # KNN has no native class_weight; handled via sample_weight in CV if needed.
        # For baseline Week 3 we use uniform weights.
        "estimator": lambda: KNeighborsClassifier(),
        "param_grid": {
            "n_neighbors": [5, 7, 9, 11, 15],
            "weights": ["uniform", "distance"],
            "metric": ["euclidean", "manhattan"],
        },
    },
    "svm": {
        "display_name": "Support Vector Machine",
        "estimator": lambda: SVC(
            class_weight="balanced",
            probability=True,  # required for predict_proba -> ROC-AUC
            random_state=RANDOM_SEED,
        ),
        "param_grid": {
            "C": [0.1, 1.0, 10.0],
            "kernel": ["rbf", "linear"],
            "gamma": ["scale", "auto"],
        },
    },
    "gradient_boosting": {
        "display_name": "Gradient Boosting",
        # GradientBoostingClassifier has no class_weight; use sample_weight via CV scorer.
        # We use subsample < 1 for stochastic GB which aids generalisation on small N.
        "estimator": lambda: GradientBoostingClassifier(
            random_state=RANDOM_SEED,
            subsample=0.8,
        ),
        "param_grid": {
            "n_estimators": [100, 200],
            "max_depth": [2, 3],
            "learning_rate": [0.05, 0.1, 0.2],
            "min_samples_leaf": [5, 10],
        },
    },
}


def get_estimator(model_key: str):
    """Return a fresh unfitted estimator for the given model key."""
    if model_key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model key '{model_key}'. Available: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[model_key]["estimator"]()


def get_param_grid(model_key: str) -> dict:
    """Return the hyperparameter grid for the given model key."""
    if model_key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model key '{model_key}'.")
    return MODEL_REGISTRY[model_key]["param_grid"]


def all_model_keys() -> list:
    return list(MODEL_REGISTRY.keys())
