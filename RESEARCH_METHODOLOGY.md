# HeartGuard AI: Machine Learning Research Methodology

**Author**: Principal ML Research Engineer & Data Scientist  
**Date**: November 2026 (Week 1 Research Phase)  
**Strict Compliance**: TRIPOD Guidelines (Item 10: Statistical analysis methods; Item 12: Internal validation; Item 13: Model building)

---

## 1. Methodological Principles & Anti-Leakage Protocol

In clinical prediction modeling, data leakage is the most frequent cause of spuriously high accuracy and subsequent external validation collapse. HeartGuard AI adheres to four non-negotiable principles:

1. **Strict Partitioning**: Preprocessing steps—including missing value imputation, standard scaling, categorical encoding, feature selection, and synthetic resampling—must be learned **exclusively on training folds** and applied transitively to validation/test folds.
2. **Encapsulation via Pipelines**: All preprocessing and modeling steps are encapsulated within `sklearn.pipeline.Pipeline` or `imblearn.pipeline.Pipeline` objects to enforce boundary isolation.
3. **Multi-Faceted Metric Evaluation**: No model will ever be selected or promoted based on classification accuracy alone. Models are judged on discrimination (ROC-AUC, PR-AUC), clinical sensitivity (recall at fixed specificity), and probability calibration (Brier score, calibration slope/intercept).
4. **Calibration Isolation**: Probability calibrators (Platt Sigmoid scaling and Isotonic regression) must be fit on dedicated validation holdouts, never on the data used to train the base estimator.

---

## 2. Model Zoo Specifications

HeartGuard AI benchmarks 10 diverse classification architectures spanning parametric, non-parametric, linear, tree-based, and ensemble families:

### 2.1 Traditional & Interpretable Baselines (5 Models)
1. **$L_2$-Regularized Logistic Regression**:
   - *Rationale*: Gold-standard clinical benchmark; provides direct odds ratios and natural probability calibration.
   - *Hyperparameters*: Inverse regularization strength $C \in [10^{-3}, 10^2]$, solver `lbfgs`.
2. **Decision Tree Classifier (CART)**:
   - *Rationale*: Inherently transparent decision boundaries simulating clinical flowcharts.
   - *Hyperparameters*: Max depth $d \in [2, 6]$, min samples leaf $\in [5, 20]$, splitting criterion (Gini vs. Entropy).
3. **Random Forest Classifier**:
   - *Rationale*: Bagged ensemble reducing variance; robust to minor non-linear interactions and tabular noise.
   - *Hyperparameters*: Estimators $N \in [100, 300]$, max features `sqrt`, max depth $\in [4, 10]$.
4. **K-Nearest Neighbors (KNN)**:
   - *Rationale*: Instance-based non-parametric baseline reflecting local patient neighborhood similarity.
   - *Hyperparameters*: Neighbors $k \in [3, 15]$, distance weights (Uniform vs. Distance), metric Euclidean/Manhattan.
5. **Support Vector Classifier (SVC)**:
   - *Rationale*: Maximum-margin hyperplane classification with RBF kernel for complex non-linear boundaries.
   - *Hyperparameters*: Regularization $C \in [0.1, 10]$, kernel parameter $\gamma$ (`scale`, `auto`, or tuned).

### 2.2 Advanced Ensembles & Gradient Boosting (5 Models)
6. **Gradient Boosting Classifier (HistGradientBoosting / sklearn GBDT)**:
   - *Rationale*: Classical sequential boosting building shallow additive regression trees.
   - *Hyperparameters*: Learning rate $\eta \in [0.01, 0.1]$, max depth $\in [3, 5]$, subsample $\in [0.8, 1.0]$.
7. **XGBoost (eXtreme Gradient Boosting)**:
   - *Rationale*: Industry-standard gradient boosted decision trees with second-order Taylor approximation and column subsampling.
   - *Hyperparameters*: `max_depth` $\in [3, 6]$, `learning_rate` $\in [0.01, 0.1]$, `subsample` $\in [0.7, 1.0]$, `colsample_bytree` $\in [0.7, 1.0]$.
8. **LightGBM (Light Gradient Boosting Machine)**:
   - *Rationale*: Leaf-wise tree growth with histogram binning; fast execution on structured tabular data.
   - *Hyperparameters*: `num_leaves` $\in [15, 31]$, `learning_rate` $\in [0.01, 0.1]$, `min_child_samples` $\in [10, 30]$.
9. **CatBoost (Categorical Boosting)**:
   - *Rationale*: Symmetric (oblivious) trees with ordered target encoding designed specifically for tabular data with categorical predictors.
   - *Hyperparameters*: `depth` $\in [4, 6]$, `learning_rate` $\in [0.01, 0.1]$, `l2_leaf_reg` $\in [1, 5]$.
10. **Stacking Classifier (Meta-Ensemble)**:
    - *Rationale*: Blends out-of-fold probabilistic predictions of top heterogeneous baselines (e.g., Logistic Regression + Random Forest + XGBoost) using regularized logistic regression as the meta-learner.

---

## 3. Experimental Evaluation Framework

```
                          Full Primary Dataset (e.g., Cleveland N=303)
                                          │
                  ┌───────────────────────┴───────────────────────┐
                  ▼                                               ▼
         Outer Stratified CV /                     Independent Test Set
         Repeated Stratified 5-Fold                      (Holdout 20%)
                  │
        ┌─────────┴─────────┐
        ▼                   ▼
  Training Fold       Validation Fold
      (80%)               (20%)
        │                   │
  [Preprocess]        [Transform Only]
  - Impute (median)         │
  - Scale (Standard)        │
  - SMOTE (train only)      │
        │                   │
  [Fit Base Model]          │
        │                   │
        ├───────────────────┘
        ▼
  [Calibrator Fitting]  (Platt / Isotonic on out-of-fold predictions)
        │
        ▼
  [Performance Evaluation]
  - ROC-AUC / PR-AUC
  - Brier Score / Calibration Slope & Intercept
  - Subgroup Disaggregated Metrics
```

### 3.1 Validation Protocols
1. **Repeated Stratified $K$-Fold Cross-Validation**:
   - $K = 5$ folds, 5 repeats (total 25 iterations) for primary model benchmark and hyperparameter optimization.
   - Stratification preserves the positive/negative outcome ratio across all folds.
2. **Nested Cross-Validation**:
   - Inner 3-fold loop for hyperparameter search via `RandomizedSearchCV` or `Optuna`.
   - Outer loop for unbiased performance estimation.
3. **External Cross-Dataset Transfer Testing (Type 3)**:
   - Train on the complete Cleveland derivation set ($N=303$).
   - Direct, zero-shot evaluation on:
     - Hungarian cohort ($N=294$)
     - Long Beach VA cohort ($N=200$)
     - Switzerland cohort ($N=123$)
   - Evaluation across harmonized feature subsets (handling missingness without leaking test labels).

---

## 4. Comprehensive Evaluation Metrics

HeartGuard AI reports 12 complementary metrics to evaluate discrimination, precision-recall trade-offs, calibration, and classification errors:

1. **Accuracy**: $\frac{TP + TN}{TP + TN + FP + FN}$ (Reported for completeness; never used for model selection).
2. **Sensitivity / Recall**: $\frac{TP}{TP + FN}$ (Critical clinical metric: proportion of diseased patients detected).
3. **Specificity**: $\frac{TN}{TN + FP}$ (Proportion of non-diseased individuals correctly spared unnecessary intervention).
4. **Positive Predictive Value (Precision)**: $\frac{TP}{TP + FP}$.
5. **Negative Predictive Value (NPV)**: $\frac{TN}{TN + FN}$.
6. **$F_1$-Score**: Harmonic mean of Precision and Recall.
7. **ROC-AUC**: Area Under the Receiver Operating Characteristic curve (ranking discrimination across all thresholds).
8. **PR-AUC**: Area Under the Precision-Recall curve (essential for imbalanced Framingham 10-year risk cohort).
9. **Confusion Matrix**: Full breakdown of $TP, FP, TN, FN$ at default ($p=0.5$) and clinically optimized thresholds.
10. **Brier Score**: Mean squared error between predicted probabilities and actual binary outcomes:
    $$\text{Brier} = \frac{1}{N} \sum_{i=1}^N (\hat{p}_i - y_i)^2$$
    (Scores range from $0.0$ [perfect] to $0.25$ [uninformative 50/50 prior]).
11. **Calibration Slope and Intercept**:
    Derived from the logistic recalibration model $\text{logit}(y) = \alpha + \beta \cdot \text{logit}(\hat{p})$.
    - Ideal Intercept $\alpha = 0$ (no systematic over/under-estimation).
    - Ideal Slope $\beta = 1$ (predictions not systematically too extreme or compressed).
12. **Bootstrap Confidence Intervals**:
    $95\%$ empirical confidence intervals computed via 1,000 non-parametric bootstrap resamples of the test set.

---

## 5. Class Imbalance Mitigation Strategy

In clinical datasets (e.g., Framingham with $\sim 15\%$ event prevalence), unweighted models become biased toward the majority class. HeartGuard AI implements a controlled comparison:
1. **Algorithmic Class Weighting**: Penalizing minority misclassification in the loss function (`class_weight='balanced'` in sklearn; `scale_pos_weight` in XGBoost/LightGBM).
2. **Synthetic Minority Oversampling (SMOTE)**:
   - Applied **strictly within the training fold** of each cross-validation iteration using `imblearn.pipeline`.
   - Never applied to test or validation folds.
   - Evaluated systematically against simple cost-sensitive weighting to verify whether synthetic samples distort feature distributions.

---

## 6. Probability Calibration Protocol

To achieve **Moderate Calibration** ([REF-09], Van Calster et al., 2016):
1. **Uncalibrated Baseline**: Extract raw predicted probabilities $\hat{p}$ from model `.predict_proba()`.
2. **Platt Scaling (Sigmoid Recalibration)**:
   Fits a logistic regression model on the model's logit outputs:
   $$\hat{p}_{\text{cal}} = \frac{1}{1 + \exp(- (a \cdot z + b))}$$
   where $z = \text{logit}(\hat{p})$. Parametric, robust on small sample sizes ($N < 500$).
3. **Isotonic Regression**:
   Fits a non-parametric, piece-wise constant monotonic step function. High expressive capacity, but prone to overfitting on small validation splits.
4. **Validation Isolation**: Calibrators are fit using 5-fold cross-validation or a dedicated calibration split, ensuring the test set evaluates post-calibration performance without optimism.

---

## 7. Explainability Protocol (SHAP)

1. **Global Attribution**:
   - Compute SHAP values across the entire validation cohort using TreeSHAP for tree ensembles and KernelSHAP for linear/kernel models.
   - Generate Beeswarm summary plots and mean absolute SHAP feature importance rankings.
2. **Local Prediction Attribution**:
   - For any individual inference query, calculate patient-specific SHAP values.
   - Render waterfall plots displaying the base value $\mathbb{E}[f(X)]$, positive risk drivers (pushing probability higher), and protective factors (pushing probability lower).
3. **Clinical Interpretation Guardrails**:
   Every visual explanation carries explicit disclaimers:
   > *"SHAP values quantify algorithmic associations within the model, not biological causes. Features flagged as high-risk reflect observed statistical patterns, not therapeutic levers."*

---

## 8. Subgroup Fairness & Out-of-Distribution Guardrails

1. **Subgroup Disaggregation**:
   - Performance metrics (Sensitivity, Specificity, Brier score) computed separately for:
     - Biological Sex: Female ($n=97$ Cleveland) vs. Male ($n=206$ Cleveland).
     - Age Groups: Younger ($<50$), Middle ($50–65$), Older ($>65$).
   - Equalized Odds disparity $\Delta_{\text{EO}} = |\text{TPR}_{\text{male}} - \text{TPR}_{\text{female}}| + |\text{FPR}_{\text{male}} - \text{FPR}_{\text{female}}|$ tracked across models.
2. **Out-of-Distribution (OOD) Guardrail Implementation**:
   - Compute multivariate empirical mean vector $\hat{\mu}$ and covariance matrix $\hat{\Sigma}$ on standardized training features.
   - For any incoming patient vector $x_{\text{new}}$, compute the squared Mahalanobis distance:
     $$D^2(x_{\text{new}}) = (x_{\text{new}} - \hat{\mu})^T \hat{\Sigma}^{-1} (x_{\text{new}} - \hat{\mu})$$
   - If $D^2(x_{\text{new}}) > \chi^2_{p, 0.99}$, flag input as an out-of-distribution anomaly.
   - Measure predictive Shannon entropy $H(\hat{p}) = - \sum_{c} \hat{p}_c \log_2 \hat{p}_c$. Inputs near maximum entropy ($H \approx 1.0$) trigger high uncertainty warnings.
