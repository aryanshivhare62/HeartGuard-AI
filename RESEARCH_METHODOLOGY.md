# HeartGuard AI: Machine Learning Research Methodology

**Author**: Principal ML Research Engineer & Data Scientist  
**Date**: November 2026 (Week 1 Research Phase)  
**Strict Compliance**: TRIPOD Statement (2015) & TRIPOD+AI Statement (Collins et al., 2024 [REF-01b])

---

## 1. Methodological Principles & Anti-Leakage Protocol

In clinical prediction modeling, data leakage is the most frequent cause of spuriously high accuracy and subsequent external validation collapse. HeartGuard AI adheres to four non-negotiable principles:

1. **Strict Partitioning**: Preprocessing steps—including missing value imputation, standard scaling, categorical encoding, feature selection, and synthetic resampling—must be learned **exclusively on training folds** and applied transitively to validation/test folds.
2. **Encapsulation via Pipelines**: All preprocessing and modeling steps are encapsulated within `sklearn.pipeline.Pipeline` or `imblearn.pipeline.Pipeline` objects to enforce boundary isolation.
3. **Multi-Faceted Metric Evaluation**: No model will ever be selected or promoted based on classification accuracy alone. Models are judged on discrimination (ROC-AUC, PR-AUC), clinical sensitivity (recall at fixed specificity), and probability calibration (Brier score, calibration slope/intercept).
4. **Calibration Isolation**: Probability calibrators (Platt Sigmoid scaling and Isotonic regression) must be fit on dedicated validation holdouts, never on the data used to train the base estimator.
5. **Frozen External Validation Boundary**: All model selection, feature engineering, hyperparameter tuning, calibrator fitting, and decision threshold determination are performed exclusively on the Cleveland cohort (or internal training folds). External test cohorts (Hungary, Long Beach, Switzerland) are strictly frozen test targets evaluated zero-shot; they must never be used for tuning or threshold adjustments.
6. **ECG Tabular Boundary**: ECG features are strictly limited to discrete tabular clinical variables (`restecg`, `oldpeak`, `slope`) as recorded on clinical case reports. Raw ECG waveform signals, digital voltage time-series, or signal filtering are explicitly excluded from the current scope.

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

### 3.1 Statistical Assessment of Validation Schemes on Small Datasets ($N=303$)

A critical methodological question in clinical biostatistics is whether **nested five-fold cross-validation** is statistically appropriate for small datasets such as the Cleveland cohort ($N=303$, with 139 positive and 164 negative cases).

#### Statistical Dilemma:
- In a standard $5 \times 5$ nested cross-validation scheme:
  - The outer training partition contains $\approx 242$ samples, and the outer test partition contains $\approx 61$ samples.
  - The inner loop further splits the $242$ samples into $80/20$ splits, yielding inner training sets of $\approx 194$ samples and inner validation sets of **only $\approx 48$ samples** (containing only $\approx 22$ positive events).
- **Statistical Consequences on $N \approx 48$ Inner Folds**:
  1. *High Estimator Variance*: A change in outcome classification for just 2 or 3 patient cases shifts fold accuracy or sensitivity by $5\%–6\%$, making hyperparameter selection noisy and unstable across outer folds.
  2. *Hyperparameter Incoherence*: Different outer folds often select divergent hyperparameter configurations (e.g., Fold 1 selects `max_depth=3`, while Fold 2 selects `max_depth=6`), meaning nested CV evaluates the *procedure* rather than producing a single final clinical model.
  3. *Small Cell Sizes in Calibration*: Evaluating probability calibration curves across $48$ samples yields severely underpowered decile bins ($4–5$ patients per bin), rendering calibration assessment inside inner folds unreliable.

#### HeartGuard AI Methodological Decision:
1. **Primary Internal Validation**: Use **Repeated Stratified $K$-Fold Cross-Validation** ($K=5$, 10 repeats, total 50 evaluations) for model benchmarking and stability analysis. Hyperparameter tuning will use constrained search spaces with strong $L_2$ shrinkage and shallow tree depths to prevent overfitting.
2. **Empirical Uncertainty Quantification**: All internal performance estimates will be accompanied by empirical $95\%$ confidence intervals derived from non-parametric bootstrapping ($B=1,000$ iterations) and DeLong's variance test for ROC-AUC.
3. **Explicit Documentation of Small Sample Limitations**: We explicitly document that on $N=303$, statistical power to detect small differences in AUC ($\Delta\text{AUC} < 0.03$) between competitive models is limited.

---

### 3.2 External Cross-Dataset Transfer Protocol (Strictly Frozen)

```
       Cleveland Primary Derivation Cohort (N=303)
                           │
             [Model Development & Tuning]
             - Repeated Stratified 5-Fold CV
             - Hyperparameter Selection
             - Platt / Isotonic Calibrator Fit
             - Decision Threshold Optimization (tau)
                           │
             ══════════════╪════════════════════════════════════════════
                           │   FROZEN MODEL ARTIFACT (Zero Leakage)
                           ▼   (Weights, Imputers, Calibrators, Thresholds Fixed)
             ┌─────────────┼─────────────┐
             ▼             ▼             ▼
       Hungarian (N=294) Long Beach (N=200) Switzerland (N=123)
       [Zero-Shot]       [Zero-Shot]        [Zero-Shot]
```

- **Protocol Rules**:
  1. No tuning, feature selection, threshold adjustment, or retraining is permitted on Hungary, Long Beach, or Switzerland.
  2. Missing feature entries in external sets (e.g., zero cholesterol) are transformed strictly using the frozen imputer statistics derived from Cleveland.
  3. Performance drops ($\Delta\text{AUC}$, $\Delta\text{Brier}$, $\Delta\text{ECE}$) are computed as unadjusted empirical transfer gaps.

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

## 8. Subgroup Fairness & Out-of-Distribution (OOD) Guardrails

### 8.1 Qualified Demographic Fairness Evaluation
- **Premise**: Subgroup performance must be analyzed disaggregated by Biological Sex (Female vs. Male) and Age strata ($<50$, $50–65$, $>65$).
- **Scientific Qualification on Fairness Impossibility**:
  - As proven by biostatistical literature (e.g., Rajkomar et al., 2018 [REF-16]), when baseline disease prevalence differs across demographic groups (in Cleveland: male prevalence $\approx 55.3\%$, female prevalence $\approx 25.8\%$), **it is mathematically impossible to simultaneously satisfy Equalized Odds (equal TPR and FPR) and Predictive Parity (equal PPV/Precision)** unless the classifier achieves $100\%$ accuracy.
  - Therefore, HeartGuard AI does not claim that a single "fairness optimizer" can eliminate all disparities. Instead, we conduct transparent audit reporting, documenting sensitivity, specificity, and positive predictive value disaggregated across demographic strata.

---

### 8.2 Out-of-Distribution (OOD) Detection Methodology & Limitations

#### 1. Methodological Formulation:
To detect patient vectors that lie far outside the clinical data distribution of the training cohort, we compute the squared Mahalanobis distance:
$$D^2(x) = (x - \hat{\mu})^T \hat{\Sigma}_{\text{shrink}}^{-1} (x - \hat{\mu})$$
where $\hat{\mu}$ is the sample mean vector and $\hat{\Sigma}_{\text{shrink}}$ is a regularized covariance matrix.

#### 2. Regularized Shrinkage Covariance (Ledoit-Wolf):
- **Limitation of Empirical Covariance**: On moderate sample sizes with correlated clinical variables (e.g., systolic BP, age, cholesterol), standard sample covariance matrices $\hat{\Sigma}$ can become ill-conditioned or nearly singular, leading to unstable matrix inversion $\hat{\Sigma}^{-1}$ and erratic distance spikes.
- **Solution**: We implement **Ledoit-Wolf shrinkage** covariance estimation (`sklearn.covariance.LedoitWolf`), which shrinks the sample covariance toward an identity-scaled target:
  $$\hat{\Sigma}_{\text{shrink}} = (1 - \lambda) \hat{\Sigma} + \lambda \mu_{\text{trace}} I$$
  where $\lambda \in [0, 1]$ is analytically selected to minimize the mean squared error of the covariance estimator.

#### 3. Limitations of Mahalanobis Distance & Prediction Entropy:
- **Multivariate Gaussianity Assumption**: Mahalanobis distance strictly assumes that feature vectors follow a multivariate Gaussian distribution. For discrete, skewed, or multimodal clinical features (e.g., categorical chest pain `cp` or binary fasting sugar `fbs`), this assumption is violated. Thus, Mahalanobis distance is computed primarily on standardized continuous predictors (`age`, `trestbps`, `chol`, `thalach`, `oldpeak`).
- **Overconfidence of Prediction Entropy**: Prediction entropy $H(\hat{p}) = - \sum_{c} \hat{p}_c \log_2 \hat{p}_c$ is derived from model output probabilities. However, uncalibrated deep models or unregularized boosting trees can output highly confident, low-entropy predictions even for inputs far outside the training manifold. Prediction entropy must therefore be treated as an auxiliary diagnostic, never a standalone OOD detector.

#### 4. Empirical OOD Evaluation on Shifted Data:
Rather than relying solely on theoretical thresholds, the OOD detector will be evaluated empirically by testing its ability to distinguish:
1. In-distribution held-out Cleveland test cases;
2. Natural distribution shifts provided by the external hospital cohorts (e.g., Switzerland with severe inpatient spectrum bias and Long Beach VA with older veteran demographics);
3. Synthetic perturbed vectors representing physiologically implausible or extreme values (e.g., blood pressure $> 260\text{ mmHg}$, cholesterol $> 700\text{ mg/dL}$).

