# HeartGuard AI: Machine Learning Research Methodology

**Author**: Principal ML Research Engineer & Data Scientist  
**Date**: November 2026 (Week 1 Research Phase)  
**Strict Compliance**: TRIPOD Statement (2015) & TRIPOD+AI Statement (Collins et al., 2024 [REF-01b])

---

## 1. Methodological Principles & Anti-Leakage Protocol

In clinical prediction modeling, data leakage and methodological opacity are the primary causes of over-optimistic performance and subsequent external validation failure. HeartGuard AI adheres to the following non-negotiable principles:

1. **Strict Partitioning & Encapsulation**: All data-dependent transformations—including missing value imputation, scaling, categorical encoding, and synthetic resampling—must be learned **exclusively on training folds** and applied transitively to validation folds using `sklearn.pipeline.Pipeline` or `imblearn.pipeline.Pipeline`.
2. **One Coherent Internal Validation Strategy**: To avoid the statistical ambiguity of nested holdouts on small clinical samples ($N=303$), internal evaluation on Cleveland is conducted using **Repeated Stratified 5-Fold Cross-Validation** (10 repeats, total 50 iterations). No separate internal holdout is split off prior to cross-validation; external generalization is evaluated strictly on the independent hospital cohorts.
3. **Modality-Specific Preprocessing**: Preprocessing pipelines strictly respect variable data types:
   - *Continuous Numeric*: Median imputation + StandardScaler (fit strictly on training folds).
   - *Categorical Nominal*: Most-frequent imputation + OneHotEncoder (`drop='first'`, `handle_unknown='ignore'`).
   - *Binary Features*: Most-frequent imputation + passthrough (binary 0/1 preserved).
4. **Prohibition of Ordinary SMOTE After One-Hot Encoding**: Ordinary SMOTE must never be applied after one-hot encoding, as Euclidean interpolation between dummy indicators produces invalid fractional values. Cost-sensitive class weighting is the primary imbalance mitigation strategy; SMOTENC is restricted to exploratory training-fold experiments.
5. **Cross-Fitted Probability Calibration**: Probability calibrators (Platt Sigmoid and Isotonic) are fitted strictly on out-of-fold validation predictions using `CalibratedClassifierCV(cv=5)`. Base estimators and calibrators are frozen on Cleveland before zero-shot external evaluation.
6. **Frozen Decision Thresholds**: Any clinical decision threshold $\tau^*$ optimized on Cleveland (e.g., via Youden's $J$ statistic on out-of-fold predictions) is completely frozen before zero-shot external evaluation. Thresholds are never re-tuned on external cohorts.
7. **ECG Tabular Modality Boundary**: ECG information is strictly restricted to discrete tabular clinical variables (`restecg`, `oldpeak`, `slope`). Raw waveform time-series and deep 1D/2D signal models are out of scope.

---

## 2. Model Zoo Specifications

HeartGuard AI benchmarks 10 diverse classification architectures spanning linear, distance-based, tree-based, and ensemble families:

### 2.1 Traditional & Interpretable Baselines (5 Models)
1. **$L_2$-Regularized Logistic Regression**:
   - *Rationale*: Clinical gold-standard linear benchmark; provides direct log-odds interpretation.
   - *Hyperparameters*: Inverse regularization strength $C \in [10^{-3}, 10^2]$ (log scale), penalty `l2`, solver `lbfgs`, `class_weight='balanced'`.
2. **Decision Tree Classifier (CART)**:
   - *Rationale*: Highly transparent, rule-based decision logic simulating clinical flowcharts.
   - *Hyperparameters*: `max_depth` $\in [2, 5]$, `min_samples_leaf` $\in [5, 20]$, `criterion` $\in ['gini', 'entropy']$, `class_weight='balanced'`.
3. **Random Forest Classifier**:
   - *Rationale*: Bagged ensemble reducing variance; provides baseline feature importance.
   - *Hyperparameters*: `n_estimators` $\in [100, 300]$, `max_depth` $\in [3, 8]$, `max_features` $\in ['sqrt', 'log2']$, `min_samples_leaf` $\in [2, 10]$, `class_weight='balanced'`.
4. **K-Nearest Neighbors (KNN)**:
   - *Rationale*: Non-parametric instance-based baseline evaluating local clinical similarity.
   - *Hyperparameters*: `n_neighbors` $\in [3, 15]$ (odd integers), `weights` $\in ['uniform', 'distance']$, `metric` $\in ['euclidean', 'manhattan']$.
5. **Support Vector Classifier (SVC)**:
   - *Rationale*: Maximum-margin hyperplane with non-linear RBF projection.
   - *Hyperparameters*: Regularization $C \in [0.1, 10.0]$, `kernel` $\in ['rbf', 'linear']$, `gamma` $\in ['scale', 'auto']$, `class_weight='balanced'`. (Calibrated via `CalibratedClassifierCV`).

### 2.2 Advanced Ensembles & Gradient Boosting (5 Models)
6. **Gradient Boosting Classifier (sklearn GBDT / HistGradientBoosting)**:
   - *Rationale*: Classical sequential boosting building shallow additive regression trees.
   - *Hyperparameters*: `learning_rate` $\in [0.01, 0.1]$, `max_depth` $\in [2, 4]$, `n_estimators` $\in [100, 200]$, `subsample` $\in [0.7, 1.0]$.
7. **XGBoost (eXtreme Gradient Boosting)**:
   - *Rationale*: Regularized gradient boosting with second-order Taylor loss expansion and column subsampling.
   - *Hyperparameters*: `n_estimators` $\in [100, 200]$, `max_depth` $\in [2, 5]$, `learning_rate` $\in [0.01, 0.1]$, `subsample` $\in [0.7, 1.0]$, `colsample_bytree` $\in [0.7, 1.0]$, `reg_lambda` $\in [1.0, 5.0]$, `scale_pos_weight` tuned to prevalence.
8. **LightGBM (Light Gradient Boosting Machine)**:
   - *Rationale*: Highly efficient histogram-based leaf-wise tree growth.
   - *Hyperparameters*: `n_estimators` $\in [100, 200]$, `num_leaves` $\in [7, 31]$, `learning_rate` $\in [0.01, 0.1]$, `min_child_samples` $\in [10, 30]$, `subsample` $\in [0.7, 1.0]$, `class_weight='balanced'`.
9. **CatBoost (Categorical Boosting)**:
   - *Rationale*: Symmetric oblivious trees with native ordered target encoding for categorical features.
   - *Hyperparameters*: `iterations` $\in [100, 300]$, `depth` $\in [3, 6]$, `learning_rate` $\in [0.01, 0.1]$, `l2_leaf_reg` $\in [1, 5]$, `auto_class_weights='Balanced'`.
10. **Stacking Classifier (Meta-Ensemble)**:
    - *Rationale*: Blends out-of-fold probabilistic predictions from heterogeneous baselines (Logistic Regression + Random Forest + XGBoost).
    - *Meta-Learner*: $L_2$-penalized Logistic Regression with internal 5-fold cross-validation.

---

## 3. Experimental Evaluation Framework

```
══════════════════════════════════════════════════════════════════════════════
               PRIMARY DERIVATION COHORT: CLEVELAND CLINIC (N=303)
══════════════════════════════════════════════════════════════════════════════
                                     │
         Repeated Stratified 5-Fold Cross-Validation (10 Repeats = 50 Splits)
                                     │
        ┌────────────────────────────┴────────────────────────────┐
        ▼ (80% ~ 242 cases)                                       ▼ (20% ~ 61 cases)
  TRAINING FOLD                                             VALIDATION FOLD
        │                                                         │
  [Fit Preprocessor]                                        [Transform Only]
  - Numeric: Median Imputer + StandardScaler                      │
  - Categorical: Mode Imputer + OneHotEncoder                     │
  - Binary: Mode Imputer + Passthrough                            │
        │                                                         │
  [Class Imbalance: class_weight='balanced']                      │
        │                                                         │
  [Fit Base Estimator] ────────► Out-of-Fold Predictions ─────────┘
        │                                │
        │                                ▼
        │                 [Fit CalibratedClassifierCV]
        │                 (Platt Sigmoid / Isotonic on OOF)
        │                                │
        │                                ▼
        │                 [Evaluate Internal Performance]
        │                 - ROC-AUC / PR-AUC + 95% Bootstrap CI
        │                 - Calibration Slope, Intercept, Brier
        │                 - Optimize Frozen Decision Threshold tau*
        │                                │
        └────────────────────────────────┘
                         │
                         ▼
             ══════════════════════════════════════════════════════
                 FINAL FROZEN PIPELINE (Zero Parameter Leakage)
                 - Preprocessor weights frozen
                 - Model coefficients/trees frozen
                 - Calibrator mapping frozen
                 - Decision threshold tau* frozen
             ══════════════════════════════════════════════════════
                         │
        ┌────────────────┼────────────────┐
        ▼ (Zero-Shot)    ▼ (Zero-Shot)    ▼ (Zero-Shot)
  HUNGARIAN (N=294)   VA LONG BEACH (N=200) SWITZERLAND (N=123)
  [10- & 9-Feature]   [10- & 9-Feature]    [9-Feature Set]
```

### 3.1 Primary Internal Validation Strategy (Repeated Stratified 5-Fold CV)
- **Elimination of Confusing 20% Holdout**: Carving out a single 20% holdout ($N \approx 60$) from a total sample of $N=303$ leaves only $\sim 28$ positive cases in the test split, causing high sample variance and wasting statistical power.
- **Adopted Protocol**: Repeated Stratified 5-Fold Cross-Validation (10 repeats, 50 distinct train/validation splits) on the entire Cleveland cohort.
- **Hyperparameter Optimization**: Tuned within each training fold using an internal 3-fold cross-validation grid/randomized search. Parameter search grids are constrained with strong regularization to prevent fold overfitting.
- **Statistical Uncertainty**: Performance metrics are aggregated across all 50 splits, reporting the mean, standard deviation, and empirical $95\%$ non-parametric bootstrap confidence intervals ($B=1,000$ resamples).

### 3.2 External Zero-Shot Transfer Protocol
- **Frozen Model Execution**: The final production model is fit on the full Cleveland derivation cohort (with cross-fitted calibration).
- **No External Adaptation**: When evaluating on Hungary, Switzerland, and Long Beach VA:
  - No hyperparameters are re-tuned.
  - No features are re-selected.
  - No calibrators are re-fit.
  - No decision thresholds are re-optimized.
- **Evaluation Across Configurations**:
  - *Full Diagnostic Set (13 features)*: Internal Cleveland benchmark only.
  - *Common Reduced Diagnostic Set (10 features)*: Zero-shot on Hungary and Long Beach VA (handling missing cholesterol via Cleveland training median).
  - *Common Reduced Diagnostic Set (9 features)*: Zero-shot across all sites, specifically evaluating Switzerland where cholesterol was unmeasured.

---

## 4. Comprehensive Evaluation Metrics

We report 12 complementary metrics, strictly separating **discrimination** (ability to rank risks) from **calibration** (accuracy of numerical probabilities):

### 4.1 Discrimination & Threshold-Dependent Metrics
1. **ROC-AUC (Concordance Index / C-statistic)**: Area Under the Receiver Operating Characteristic curve; measures pairwise ranking accuracy across all thresholds.
2. **PR-AUC (Average Precision)**: Area Under the Precision-Recall curve; essential for evaluating models under class imbalance.
3. **Sensitivity / Recall ($\text{TPR}$)**: $\frac{TP}{TP + FN}$ (Evaluated at default $\tau=0.5$ and frozen optimal $\tau^*$).
4. **Specificity ($\text{TNR}$)**: $\frac{TN}{TN + FP}$.
5. **Positive Predictive Value (PPV / Precision)**: $\frac{TP}{TP + FP}$.
6. **Negative Predictive Value (NPV)**: $\frac{TN}{TN + FN}$.
7. **$F_1$-Score**: Harmonic mean of Precision and Recall.
8. **Accuracy**: $\frac{TP + TN}{N}$ (Reported for reporting completeness; barred from model selection).

### 4.2 Probability Calibration Metrics
9. **Brier Score**: Mean squared error between predicted probabilities $\hat{p}_i \in [0, 1]$ and binary outcomes $y_i \in \{0, 1\}$:
   $$\text{Brier} = \frac{1}{N} \sum_{i=1}^N (\hat{p}_i - y_i)^2$$
   (Reference: uninformative prevalence prior Brier $= \bar{y}(1 - \bar{y})$).
10. **Calibration Slope ($\beta$)**: Slope from logistic recalibration model $\text{logit}(y) = \alpha + \beta \cdot \text{logit}(\hat{p})$. Target $\beta = 1.0$; $\beta < 1.0$ indicates overconfident/overfit predictions.
11. **Calibration-in-the-Large (Intercept $\alpha$)**: Intercept from logistic recalibration with slope fixed to 1 ($\text{offset} = \text{logit}(\hat{p})$). Target $\alpha = 0$; $\alpha > 0$ indicates systematic underestimation of risk.
12. **Expected Calibration Error (ECE)**: Weighted average difference between predicted confidence and empirical event frequency across $M=10$ equal-width probability bins $B_m$:
    $$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$

---

## 5. Class Imbalance Strategy & Prohibition of Ordinary SMOTE

1. **Primary Imbalance Strategy (Cost-Sensitive Weighting)**:
   - Primary model training handles class imbalance via loss function re-weighting (`class_weight='balanced'` in scikit-learn, `scale_pos_weight` in XGBoost/LightGBM).
   - This approach adjusts loss gradients directly without perturbing the empirical feature manifold.
2. **Methodological Prohibition of Post-Encoding SMOTE**:
   - Applying standard SMOTE after one-hot encoding interpolates Euclidean distances between binary dummy vectors (e.g., resulting in non-binary values such as `cp_2 = 0.43`), violating categorical semantics.
3. **Exploratory SMOTENC Protocol**:
   - If synthetic oversampling is explored in secondary analyses, it must use **SMOTENC** (`imblearn.over_sampling.SMOTENC`), with explicit indices for categorical features (`cp`, `restecg`, `slope`, `thal`).
   - SMOTENC must be fitted **strictly inside training folds before one-hot encoding**. It is never applied to validation or external test sets.

---

## 6. Cross-Fitted Probability Calibration Protocol

To achieve **Moderate Calibration** ([REF-09], Van Calster et al., 2016):
1. **Cross-Fitted Post-Hoc Calibrators**:
   - Implemented via `CalibratedClassifierCV(base_estimator, cv=5, method='sigmoid' | 'isotonic')`.
   - The base classifier is fit on training subsets, and calibrator parameters (Platt logistic parameters $a, b$ or isotonic step functions) are fit on out-of-fold predictions.
2. **Platt Scaling (Sigmoid)**:
   - Parametric logistic recalibration: $\hat{p}_{\text{cal}} = \frac{1}{1 + \exp(-(a \cdot \text{logit}(\hat{p}) + b))}$. Robust and resistant to overfitting on small validation samples ($N \approx 60$).
3. **Isotonic Regression**:
   - Monotonic step-wise non-parametric fitting. Highly flexible, but susceptible to step-interval overfitting on small clinical datasets.
4. **External Deployment**:
   - The final calibrator fit on Cleveland out-of-fold predictions is frozen and evaluated zero-shot on external datasets.

---

## 7. Explainability Methodology (TreeSHAP vs. Calibrated Output)

1. **Mathematical Grounding of TreeSHAP**:
   - TreeSHAP ([REF-12], Lundberg et al., 2018) computes exact Shapley attributions in polynomial time on the **base tree ensemble's raw output (margin / log-odds space)**:
     $$\text{margin}(x) = \phi_0 + \sum_{i=1}^M \phi_i(x)$$
   - In margin space, the efficiency and additivity axioms hold exactly.
2. **Impact of Post-Hoc Calibration on SHAP Attributions**:
   - When Platt scaling ($\hat{p} = \sigma(a \cdot \text{margin} + b)$) is appended to a tree model, the scaling function is **strictly monotonic**.
   - Consequently, the sign (positive vs. negative risk contribution) and relative rank-ordering of feature attributions are mathematically preserved from margin space.
   - For individual patient explanations, waterfall plots display attributions in base log-odds space with a secondary axis indicating monotonic calibrated risk percentage.
3. **Clinical Interpretation Disclaimer**:
   - Every SHAP output explicitly carries the statement:
     > *"SHAP values quantify algorithmic associations within the model, not biological causes. Features flagged as high-risk reflect observed statistical patterns, not therapeutic levers."*

---

## 8. Subgroup Fairness & Inherent Trade-Offs

### 8.1 Theoretical Grounding in Fairness Impossibility
As proved by **Kleinberg, Mullainathan, & Raghavan (2016/2017 [REF-16b])** and **Rajkomar et al. (2018 [REF-16])**:
- If baseline disease prevalence differs between two demographic subgroups (in Cleveland: male prevalence $\approx 55.3\%$, female prevalence $\approx 25.8\%$), **it is mathematically impossible for a non-perfect prediction model to satisfy both Equalized Odds (equal TPR and FPR) and Predictive Parity (equal PPV/Precision) simultaneously**.
- Attempting to force equal prediction rates (Demographic Parity) across groups with differing underlying true disease rates guarantees clinical harm through false-positive overtreatment in low-prevalence groups or false-negative underdiagnosis in high-prevalence groups.

### 8.2 Disaggregated Subgroup Audit Protocol
Rather than attempting contradictory mathematical parity optimization, HeartGuard AI executes a rigorous **transparency audit**:
- Metrics (Sensitivity, Specificity, PPV, Brier score, Calibration slope) are reported disaggregated by:
  - Biological Sex: Female ($N=97$) vs. Male ($N=206$).
  - Age Categories: Younger ($<50$ years), Middle ($50–65$ years), Older ($>65$ years).
- We report the observed Equalized Odds gap $|\Delta\text{TPR}| + |\Delta\text{FPR}|$ without enforcing artificial equality that degrades clinical safety.

---

## 9. Out-of-Distribution (OOD) Detection Methodology & Limitations

### 9.1 Formulation with Regularized Ledoit-Wolf Shrinkage
To detect patient vectors outside the supported clinical training distribution:
$$D^2(x) = (x - \hat{\mu})^T \hat{\Sigma}_{\text{shrink}}^{-1} (x - \hat{\mu})$$
- $\hat{\mu}$ and $\hat{\Sigma}_{\text{shrink}}$ are computed using **Ledoit-Wolf shrinkage** (`sklearn.covariance.LedoitWolf`) strictly on the training fold's standardized continuous features (`age`, `trestbps`, `chol`, `thalach`, `oldpeak`).
- Shrinkage stabilizes the empirical covariance matrix against collinearity and small sample size, ensuring well-conditioned matrix inversion.

### 9.2 Methodological Limitations
1. **Normality Assumption**: Mahalanobis distance assumes multivariate Gaussianity. Categorical and binary features are excluded from the metric to avoid violating distributional assumptions.
2. **Prediction Entropy Overconfidence**: Predictive Shannon entropy $H(\hat{p}) = - \sum \hat{p}_c \log_2 \hat{p}_c$ can be deceptively low (overconfident) for tree ensembles and neural networks on out-of-domain inputs. It is reported strictly as an auxiliary diagnostic.
3. **Empirical Evaluation**: OOD detection is evaluated empirically by testing its capacity to separate held-out Cleveland cases from natural demographic shifts (Hungary, Long Beach VA, Switzerland) and synthetic extreme vectors (e.g., blood pressure $> 250\text{ mmHg}$).

---

## 10. Separation of Confirmatory vs. Exploratory Analyses

To prevent data dredging and post-hoc rationalization, all experimental tasks are partitioned into two distinct categories:

| Category | Experimental Scope | Hypotheses & Protocol |
| :--- | :--- | :--- |
| **Primary (Confirmatory) Analysis** | - 10-model benchmark on Cleveland via Repeated Stratified 5-Fold CV.<br>- Zero-shot transfer to Hungary, Long Beach VA, and Switzerland under frozen parameters.<br>- Evaluation on Full (13), Common Reduced (10), and Common Reduced (9) feature sets.<br>- Primary metrics: ROC-AUC, PR-AUC, Brier score, Calibration slope/intercept. | Pre-specified hypotheses ([RQ1]–[RQ3]). Strict zero-leakage, frozen pipelines, bootstrap 95% CIs. |
| **Secondary (Exploratory) Analysis** | - Comparison of SMOTENC oversampling vs. cost-sensitive class weighting.<br>- Sensitivity analysis of cholesterol median imputation vs. feature dropping on Swiss cohort.<br>- Ledoit-Wolf OOD distance cutoff sensitivity across natural hospital shifts.<br>- Subgroup disparity trade-off exploration. | Hypothesis-generating; clearly designated as exploratory in all reports. |
