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

### 3.1 Primary Internal Validation Strategy & Statistical Uncertainty

#### 1. Internal Validation Protocol on Cleveland ($N=303$):
- **Elimination of Confusing 20% Holdout**: Carving out a single 20% holdout ($N \approx 60$) from a small dataset of $N=303$ leaves only $\sim 28$ positive events in the test split, causing high estimator variance and wasting precious sample size.
- **Adopted Protocol**: Repeated Stratified 5-Fold Cross-Validation (10 repeats, total 50 distinct train/validation splits) on the complete Cleveland derivation cohort.
- **Hyperparameter Optimization**: Tuned strictly within the training fold of each split (using internal 3-fold cross-validation or randomized search). Parameter grids are heavily regularized to prevent fold overfitting.

#### 2. Rigorous Statistical Uncertainty Reporting:
- **Non-Independence of Repeated CV Folds**: The 50 fold-level metric results across repeated cross-validation are **statistically dependent and correlated** because folds share training and validation instances across repeats. Therefore, treating the 50 fold scores as independent observations (e.g., dividing sample standard deviation by $\sqrt{50}$ to report standard error, or conducting naive Student's t-tests) is statistically invalid and severely underestimates uncertainty.
- **Patient-Level Non-Parametric Bootstrap CIs**: To report valid confidence intervals, bootstrap resampling must be performed at the **patient observation level**:
  - Resample the patient cohort ($N=303$) with replacement $B=1,000$ times.
  - On each bootstrap resample, evaluate model predictions and compute the evaluation metric.
  - The empirical 95% confidence interval is extracted using the percentile method ($2.5^{\text{th}}$ and $97.5^{\text{th}}$ percentiles).
- **DeLong Test Application Boundary**: DeLong's test for comparing ROC curves (DeLong et al., 1988) strictly assumes independent patient observations. **DeLong testing must NEVER be applied to predictions pooled across repeated cross-validation folds**, as repeated predictions for the same patient violate sample independence. DeLong's test is restricted exclusively to comparing competing models on a single fixed cohort (e.g., on a single holdout split or on an external validation dataset).
- **Limitations of Small Subgroup Confidence Intervals**: When evaluating demographic subgroups with very small sample sizes (e.g., female subgroup in Cleveland with $N=97$ and only $\sim 25$ events, or external Switzerland females with $N=10$), bootstrap confidence intervals become extremely wide, unstable, and sensitive to boundary effects. Subgroup metrics must always be reported alongside exact sample sizes ($N$) and event counts ($E$), and interpreted with documented caution.

---

### 3.2 External Zero-Shot Transfer Protocol (Strictly Frozen)

```
       CLEVELAND DERIVATION COHORT (N=303)
                         │
        ┌────────────────┼────────────────┐
        ▼                ▼                ▼
   Model 1 (Full)   Model 2 (10-Feat) Model 3 (9-Feat)
   13 Predictors    10 Predictors    9 Predictors
   [Trained Sep.]   [Trained Sep.]   [Trained Sep.]
        │                │                │
   [Freeze All]     [Freeze All]     [Freeze All]
        │                │                │
        ▼                ▼                ▼
   Cleveland CV     Zero-Shot on     Zero-Shot on
   Benchmark Only   Hungary & VA     All 3 External Sites
                    (Hungary N=294,  (Specifically
                     Long Beach 200)  Switzerland N=123)
```

- **Independent Model Training per Configuration**:
  - The three feature configurations—**Full Diagnostic (13 features)**, **Common Reduced (10 features, with cholesterol)**, and **Common Reduced (9 features, without cholesterol)**—are **trained and tuned as three completely separate, independent pipelines on Cleveland**.
  - A 10-feature model is trained strictly on Cleveland using only those 10 features; its preprocessing scaler, hyperparameters, calibrator, and decision threshold are optimized exclusively on that 10-dimensional space.
  - Similarly, the 9-feature model is trained strictly on Cleveland using only those 9 features.
  - Models never drop features post-hoc at test time from a 13-feature model.
- **Strictly Frozen Zero-Shot Execution**:
  - No hyperparameters, features, calibrators, decision thresholds, or OOD cutoffs are modified or re-tuned on external datasets.
  - Performance drops ($\Delta\text{ROC-AUC}$, $\Delta\text{PR-AUC}$, $\Delta\text{Brier}$) are computed as unadjusted empirical generalization gaps.

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

To achieve **Moderate Calibration** ([REF-09], Van Calster et al., 2016) without data leakage:

### 6.1 Scikit-Learn API & Exact Cross-Fitting Procedure
- **Scikit-Learn API Implementation**:
  In current scikit-learn releases ($\ge 1.2$, updated in 1.4+), probability calibration is parameterized using `estimator` (replacing the deprecated `base_estimator`):
  ```python
  from sklearn.calibration import CalibratedClassifierCV
  
  calibrated_clf = CalibratedClassifierCV(
      estimator=base_estimator,
      method='sigmoid',  # or 'isotonic'
      cv=5
  )
  ```
- **Cross-Fitting Mechanism**:
  When `cv=5`, `CalibratedClassifierCV` partitions the training data into 5 folds. For each fold $k \in \{1,\dots,5\}$:
  1. The `base_estimator` is fit on the remaining 4 folds (80% of data).
  2. Uncalibrated predictions are generated for the held-out $k^{\text{th}}$ fold (20% of data).
  3. A calibrator mapping (Platt logistic regression or isotonic step function) is fit strictly on those out-of-fold predictions.
  4. At inference time, the ensemble averages the calibrated probabilities across all 5 fold models: $\hat{p}(x) = \frac{1}{5} \sum_{k=1}^5 \hat{p}_k(x)$.
- **Strict Out-of-Fold Calibration Evaluation Rule**:
  When evaluating whether a model is properly calibrated on the Cleveland cohort, predicted probabilities must be generated strictly from observations **not** used to fit the corresponding base estimator or calibrator (e.g., using `cross_val_predict(..., method='predict_proba')`). Evaluating calibration on in-sample training predictions yields spuriously low Brier scores and falsely ideal calibration curves.
- **External Evaluation**:
  The final `CalibratedClassifierCV` ensemble fit on Cleveland is exported and applied zero-shot to external cohorts without recalibration.

---

## 7. Explainability Methodology (TreeSHAP vs. Calibrated Output)

### 7.1 Mathematical Grounding of TreeSHAP in Raw Margin Space
TreeSHAP ([REF-12], Lundberg et al., 2018) calculates exact Shapley attributions in polynomial time on the **raw output space of the base tree ensemble (log-odds / margin space)**:
$$\text{margin}(x) = \phi_0 + \sum_{i=1}^M \phi_i(x)$$
In this uncalibrated margin space, the fundamental game-theoretic axioms (Efficiency, Symmetry, Dummy, and Additivity) hold exactly.

### 7.2 Non-Equivalence in Calibrated Probability Space
> [!IMPORTANT]
> **Mathematical Correction on Calibration and Feature Attributions**:  
> A common misconception in applied machine learning is that because Platt scaling $\hat{p} = \sigma(a \cdot \text{margin} + b)$ is a monotonic function, TreeSHAP attributions computed in margin space automatically preserve feature attribution signs or proportional rankings in probability space. **This is mathematically incorrect**:
> 1. *Non-Linearity*: The sigmoid transformation $\sigma(z) = \frac{1}{1 + \exp(-z)}$ is non-linear; its derivative $\sigma'(z) = \sigma(z)(1 - \sigma(z))$ varies from near zero in the tails to $0.25$ at the center.
> 2. *Context-Dependent Marginal Effects*: In probability space, the marginal change in predicted probability attributable to feature $i$ depends heavily on the cumulative sum of all other features (i.e., where the patient is located along the sigmoid curve). A feature with a large positive attribution in log-odds space may produce a negligible change in probability if the patient is already deep in the saturated tail ($\hat{p} \approx 0.98$).
> 3. *Attribution Re-ranking*: Because of non-linear saturation, computing Shapley values directly on the composite function $\sigma(f(x))$ can alter the relative rank-ordering of feature importances compared to margin-space TreeSHAP.

### 7.3 HeartGuard AI Implementation Strategy
- **Primary Attribution Framework**: TreeSHAP waterfall and beeswarm plots are computed and presented in the **base model's uncalibrated log-odds (margin) space**, where additivity is exact and theoretically grounded.
- **Clinical Interface Presentation**: The user interface displays the exact raw log-odds contributions $\phi_i$, accompanied by an explicit clinical banner:
  > *"Feature attributions are computed in the base model's log-odds score space before non-linear calibration. Positive values indicate features that push the model toward higher cardiovascular risk; negative values indicate features that reduce risk. These attributions represent statistical associations within the retrospective model, not biological causes or direct probability percentages."*
- **Model-Agnostic Probability Explanations**: If explanations directly on the calibrated probability $\hat{p} \in [0, 1]$ are required, they are computed using model-agnostic KernelSHAP on the calibrated pipeline and explicitly designated as approximations.

---

## 8. Subgroup Fairness & Theoretical Inherent Trade-Offs

### 8.1 Rigorous Formulation of Kleinberg et al.'s Impossibility Theorem
HeartGuard AI grounds its fairness evaluation in the landmark impossibility theorem proved by **Kleinberg, Mullainathan, & Raghavan (2016/2017 [REF-16b])**:
- Consider a continuous risk score $R \in [0, 1]$ and a binary outcome $Y \in \{0, 1\}$ evaluated across demographic groups $A \in \{1, 2\}$ with differing baseline disease prevalence ($p_1 \neq p_2$, as observed between males [$\approx 55\%$] and females [$\approx 26\%$] in the Cleveland cohort).
- Kleinberg et al. investigated three specific, commonly desired fairness conditions:
  1. **Calibration within groups**: For each group $a$ and each score value $r$, $P(Y=1 \mid R=r, A=a) = r$.
  2. **Balance for the positive class**: The average score assigned to subjects with $Y=1$ is equal across groups ($\mathbb{E}[R \mid Y=1, A=1] = \mathbb{E}[R \mid Y=1, A=2]$, equivalent to equal False Negative Rates / Equal Opportunity at fixed thresholds).
  3. **Balance for the negative class**: The average score assigned to subjects with $Y=0$ is equal across groups ($\mathbb{E}[R \mid Y=0, A=1] = \mathbb{E}[R \mid Y=0, A=2]$, equivalent to equal False Positive Rates at fixed thresholds).
- **The Theorem**: If baseline group prevalence rates differ ($p_1 \neq p_2$), **no risk score can simultaneously satisfy calibration within groups, balance for the positive class, and balance for the negative class**, unless the score achieves perfect deterministic accuracy (assigning $R \in \{0, 1\}$ with zero prediction error).
- **Academic Precision Rule**: We do not misrepresent this theorem as proving that every arbitrary pair of fairness metrics is mutually incompatible; rather, it specifically proves the inherent mathematical conflict between probability calibration within groups and error-rate parity across groups when base prevalence differs.

### 8.2 Disaggregated Subgroup Audit Protocol
Because forcing error-rate balance artificially distorts probability calibration (and vice versa), HeartGuard AI rejects naive "fairness optimization" that could induce clinical miscalibration. Instead, we implement **transparent disaggregated reporting**:
- Report Sensitivity, Specificity, Positive Predictive Value, Brier score, and Calibration slope disaggregated by:
  - Biological Sex: Female ($N=97$, $E=25$) vs. Male ($N=206$, $E=114$).
  - Age Categories: Younger ($<50$), Middle ($50–65$), Older ($>65$).
- Explicitly document sample size ($N$) and event count ($E$) alongside wide bootstrap confidence intervals for small subgroups, cautioning against over-interpreting underpowered stratum metrics.

---

## 9. Out-of-Distribution (OOD) Detection Methodology & Threshold Rules

### 9.1 Formulation with Regularized Ledoit-Wolf Shrinkage
To detect patient vectors lying outside the supported training distribution:
$$D^2(x) = (x - \hat{\mu})^T \hat{\Sigma}_{\text{shrink}}^{-1} (x - \hat{\mu})$$
- $\hat{\mu}$ and $\hat{\Sigma}_{\text{shrink}}$ are computed using **Ledoit-Wolf shrinkage** (`sklearn.covariance.LedoitWolf`) strictly on the standardized continuous features (`age`, `trestbps`, `chol`, `thalach`, `oldpeak`) of the **Cleveland derivation cohort**.
- Shrinkage stabilizes the empirical covariance against collinearity, ensuring well-conditioned matrix inversion.

### 9.2 Strict Rule on Threshold Selection (Cleveland Only)
> [!CAUTION]
> **OOD Threshold Selection Boundary**:  
> The critical threshold for flagging an input as out-of-distribution (e.g., the 95th or 99th percentile $\chi^2$ cutoff) **must be determined strictly from the Cleveland training/validation data**.  
> **Under no circumstances will OOD thresholds be tuned, calibrated, or adjusted using the external cohorts (Hungary, Switzerland, or Long Beach VA).**  
> External datasets serve exclusively as natural-shift test cases to evaluate how frequently the frozen Cleveland-derived threshold flags external patient populations.

### 9.3 Methodological Limitations of OOD Detection
1. **Normality Assumption**: Mahalanobis distance assumes that the continuous features follow a multivariate Gaussian distribution. Categorical (`cp`, `restecg`, `slope`, `thal`) and binary (`sex`, `fbs`, `exang`) variables are excluded from the Mahalanobis calculation to avoid distributional invalidity.
2. **Prediction Entropy Overconfidence**: Predictive Shannon entropy $H(\hat{p}) = - \sum \hat{p}_c \log_2 \hat{p}_c$ can be deceptively low (overconfident) for tree ensembles and neural networks on out-of-domain inputs. Entropy is treated strictly as an auxiliary diagnostic, never as a primary OOD detector.

---

## 10. Separation of Confirmatory vs. Exploratory Analyses

To prevent post-hoc hypothesis generation from being mistaken for rigorous testing, all project analyses are partitioned into two distinct categories:

| Category | Experimental Scope | Hypotheses & Validation Protocol |
| :--- | :--- | :--- |
| **Primary (Confirmatory) Analysis** | - 10-model benchmark on Cleveland via Repeated Stratified 5-Fold CV.<br>- Zero-shot transfer to Hungary, Long Beach VA, and Switzerland under frozen parameters.<br>- Separate training and evaluation of Full (13), Common Reduced (10), and Common Reduced (9) feature sets.<br>- Primary metrics: ROC-AUC, PR-AUC, Brier score, Calibration slope/intercept. | Pre-specified hypotheses ([RQ1]–[RQ3]). Strict zero-leakage, frozen pipelines, patient-level bootstrap 95% CIs. |
| **Secondary (Exploratory) Analysis** | - Comparison of SMOTENC oversampling vs. cost-sensitive class weighting inside training folds.<br>- Sensitivity analysis of cholesterol median imputation vs. 9-feature omission on external cohorts.<br>- Empirical evaluation of frozen Cleveland OOD threshold sensitivity across external natural hospital shifts.<br>- Exploration of subgroup calibration vs. error-rate balance trade-offs. | Hypothesis-generating; clearly designated as exploratory in all reports and publications. |
