# RoadSafe AI: Comprehensive Technical Manual & Module-by-Module Guide

**Project Title:** RoadSafe: AI-Based Road Accident Severity Prediction and Traffic Risk Analysis System  
**Academic Level:** End-Semester Machine Learning Course Project Evaluation  
**Curriculum Alignment:** Full Coverage of Course Outcomes **CO1, CO2, CO3, CO4, CO5, and CO6**  
**Core Dataset:** US Accidents Dataset (7.7 Million+ Records, March 2023 Release)  

---

## Table of Contents
1. [Project Overview & Core Mission](#1-project-overview--core-mission)
2. [End-to-End System Architecture (CO1)](#2-end-to-end-system-architecture-co1)
3. [Deep-Dive: Module-by-Module Explanation](#3-deep-dive-module-by-module-explanation)
   - [3.1 `src/data_preprocessing.py`](#31-srcdata_preprocessingpy)
   - [3.2 `src/co2_linear_models.py`](#32-srcco2_linear_modelspy)
   - [3.3 `src/co3_tree_models.py`](#33-srcco3_tree_modelspy)
   - [3.4 `src/co4_unsupervised.py`](#34-srcco4_unsupervisedpy)
   - [3.5 `src/co5_evaluation_tuning.py`](#35-srcco5_evaluation_tuningpy)
   - [3.6 `src/co6_mlops_monitoring.py`](#36-srcco6_mlops_monitoringpy)
   - [3.7 `src/inference_pipeline.py`](#37-srcinference_pipelinepy)
   - [3.8 `app.py` (REST API & Web Controller)](#38-apppy-rest-api--web-controller)
   - [3.9 Frontend Architecture: `templates/index.html`, `style.css`, `app.js`](#39-frontend-architecture)
   - [3.10 Orchestration & Validation Scripts](#310-orchestration--validation-scripts)
4. [Course Outcomes (CO) Rigorous Verification Matrix](#4-course-outcomes-co-rigorous-verification-matrix)
5. [Mathematical & Theoretical Foundations](#5-mathematical--theoretical-foundations)
6. [Empirical Benchmark Results](#6-empirical-benchmark-results)
7. [How to Run and Present for Evaluation](#7-how-to-run-and-present-for-evaluation)

---

## 1. Project Overview & Core Mission
Motor vehicle collisions are high-impact events resulting in traffic congestion, economic losses, and life-threatening emergencies. Emergency medical personnel and traffic management agencies require immediate situational awareness to allocate medical trauma resources, deploy heavy wreckers, and initiate dynamic arterial detours.

**RoadSafe** is an applied machine learning system that:
1. **Classifies Multi-Class Accident Severity (1 to 4):** Predicts whether an incident is minor (1), lane-obstructing (2), multi-lane hazard (3), or catastrophic highway closure (4).
2. **Estimates Impact Duration (Regression):** Quantifies roadway clearance time in minutes using regularized linear estimators.
3. **Identifies High-Risk Spatial Corridors & Hotspots:** Applies unsupervised density and centroid clustering to uncover high-frequency collision segments.
4. **Maintains Complete Request Traceability (CO1):** Audits every active prediction backwards through all 6 engineering lifecycle gates.
5. **Guarantees Production MLOps Integrity (CO6):** Uses a centralized Feature Store to avoid training-serving skew and executes continuous two-sample Kolmogorov-Smirnov drift tests.

---

## 2. End-to-End System Architecture (CO1)

```
[ Historical US Accidents: 7.7M Records ]
                   │
                   ▼
  [ scripts/extract_sample.py ] ──► Stratified Sampling (80,000 Records, Classes 1-4)
                   │
                   ▼
  [ src/data_preprocessing.py ] ──► ColumnTransformer (StandardScaler + OneHotEncoder)
                   │
       ┌───────────┼───────────────────────────┐
       ▼           ▼                           ▼
  [ CO2 Linear ] [ CO3 Trees ]           [ CO4 Unsupervised ]
  • OLS / Ridge  • Decision Trees        • K-Means (Elbow, Sil)
  • Lasso (49%)  • Random Forest (Bag)   • DBSCAN (Corridors)
  • ElasticNet   • XGBoost / LightGBM    • PCA Scree / IsoForest
       └───────────┼───────────────────────────┘
                   ▼
  [ src/co5_evaluation_tuning.py ]
  • Stratified 5-Fold CV (90.6% Acc)
  • Platt/Sigmoid Calibration (Brier: 0.0601)
  • Hyperparameter Search (Grid vs Random vs Bayesian)
                   │
                   ▼
  [ src/co6_mlops_monitoring.py & src/inference_pipeline.py ]
  • Feature Store Catalog (feature_store.json)
  • Model Packaging (joblib pipeline artifacts)
  • Live Request Provenance Engine (CO1 Trace)
  • Two-Sample KS-Test Drift Detection
                   │
                   ▼
  [ app.py ] ──► Production Flask REST Server (Port 5000)
                   │
                   ▼
  [ User Browser ] ──► Interactive Web UI (HTML5, Modern CSS, Dynamic JavaScript)
```

---

## 3. Deep-Dive: Module-by-Module Explanation

### 3.1 `src/data_preprocessing.py`
#### What it is:
The core data transformation and feature pipeline engine of the project.
#### Key Components:
- **`NUMERICAL_FEATURES`**: Continuous parameters: `Temperature(F)`, `Humidity(%)`, `Pressure(in)`, `Visibility(mi)`, `Wind_Speed(mph)`, `Precipitation(in)`, `Hour`, `Month`, `Start_Lat`, `Start_Lng`.
- **`CATEGORICAL_FEATURES`**: Nominal variables: `Weather_Condition`, `Sunrise_Sunset`, `State_Clean`.
- **`BINARY_FEATURES`**: Structural roadway attributes: `Crossing`, `Give_Way`, `Junction`, `No_Exit`, `Railway`, `Roundabout`, `Station`, `Stop`, `Traffic_Calming`, `Traffic_Signal`, `Amenity`, `Bump`, `Is_Weekend`, `Rush_Hour`.
- **`create_preprocessor()`**: Constructs a scikit-learn `ColumnTransformer` composed of:
  - Numerical pipeline: `SimpleImputer(strategy='median')` followed by `StandardScaler()`.
  - Categorical pipeline: `SimpleImputer(strategy='most_frequent')` followed by `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`.
  - Binary attributes: `'passthrough'`.
- **`get_train_val_test_data()`**: Performs a stratified two-stage split:
  - 70% Training Partition (56,000 records)
  - 15% Validation Partition (12,000 records)
  - 15% Test Partition (12,000 records)
  - Strictly preserves relative class frequencies across Severity 1, 2, 3, and 4.
#### What it does for the project:
Guarantees **zero data leakage**. Scaling and imputation parameters ($\mu, \sigma, \text{median}$) are learned solely on the training partition and frozen into the transformer. When serialized, it enforces **identical transformations** at inference time.

---

### 3.2 `src/co2_linear_models.py`
#### What it is:
Implements regularized linear supervised-learning models for both regression and classification, accompanied by formal mathematical reasoning.
#### Key Models & Algorithms:
1. **Clearance Duration Regression (Continuous Output in Minutes):**
   - **`LinearRegression()` (OLS):** Serves as the unregularized empirical benchmark.
   - **`Ridge(alpha=10.0)` ($L_2$ Regularization):** Shrinks coefficient magnitudes proportionally via quadratic penalty $\alpha \sum w_j^2$. Stabilizes solutions under multicollinear weather attributes.
   - **`Lasso(alpha=0.1)` ($L_1$ Regularization):** Constrains absolute coefficient magnitudes $\alpha \sum |w_j|$. Induces **49.0% parameter sparsity (24 out of 49 weights zeroed out)**, performing embedded feature selection.
   - **`ElasticNet(alpha=0.1, l1_ratio=0.5)`:** Convex hybrid combination balancing $L_1$ sparsity and $L_2$ collinear grouping.
2. **Accident Severity Classification:**
   - **Binary Logistic Regression:** Classifies high-risk ($\ge 3$) vs low-risk ($\le 2$) collisions.
   - **Multinomial Logistic Regression (Softmax):** Models multi-class probabilities across Severity 1 to 4 via normalized exponential logits.
#### Theoretical Principles Documented:
- **Bias-Variance Trade-Off:** OLS has low training bias but higher variance. Ridge/Lasso introduce controlled shrinkage bias $\alpha$, reducing estimation variance and improving test generalization.
- **Feature Scaling Necessity:** Regularization penalties treat all weights uniformly. Without scaling, variables with large numbers (like Pressure $\approx 30$) dominate the penalty over tiny variables (Precipitation $\approx 0.1$). `StandardScaler` is required.

---

### 3.3 `src/co3_tree_models.py`
#### What it is:
Tree-based supervised learning engine implementing individual trees, bagging ensembles, and gradient boosting algorithms.
#### Key Models & Algorithms:
1. **Decision Tree (Gini Impurity vs. Shannon Entropy):**
   - Compares splitting criteria: $\text{Gini} = 1 - \sum p_i^2$ vs $\text{Entropy} = -\sum p_i \log_2 p_i$.
   - Confirms that while both achieve near-identical accuracy (83.6% vs 82.5%), Gini executes ~15% faster by avoiding logarithmic operations.
2. **Random Forest Classifier (Bagging Variance Reduction):**
   - Combines $B=100$ decorrelated decision trees using bootstrap resamples and randomized feature subspaces ($m = \sqrt{p}$).
   - Lowers pairwise correlation $\rho$, driving down ensemble variance without increasing individual tree bias.
3. **XGBoost Classifier (Extreme Gradient Boosting):**
   - Employs second-order Taylor expansion of the loss function with column subsampling and shrinkage.
4. **LightGBM Classifier (Leaf-Wise Histogram Boosting):**
   - Fast histogram binning with leaf-wise (best-first) tree growth.
   - Achieves the project's highest classification performance: **91.77% Accuracy, 0.9880 ROC-AUC (OvR)**.
#### Feature Importance Analysis:
Extracts Mean Decrease in Impurity (MDI / Gain) identifying top predictive hazard factors:
1. `Distance(mi)` (extent of road impacted)
2. `Precipitation(in)` (reduced friction)
3. `Visibility(mi)` (visual impairment)
4. `Junction` (traffic conflict points)
5. `Temperature(F)` (ice/freeze surface hazards)

---

### 3.4 `src/co4_unsupervised.py`
#### What it is:
The unsupervised learning, spatial hotspot clustering, and anomaly detection module.
#### Key Techniques:
1. **K-Means Spatial Clustering:**
   - Evaluates $K \in [3, 8]$ using the **Elbow Method (Inertia)** and **Silhouette Analysis**.
   - Discovers **5 regional accident clusters** across the US with distinct geographical centroids and risk levels:
     - West Coast Corridor (Lat 35.8, Lng -119.5)
     - Midwest Industrial Corridor (Lat 41.2, Lng -84.3)
     - Southeast / Florida (Lat 29.5, Lng -81.8)
     - Northeast Megalopolis (Lat 41.1, Lng -74.8)
     - Texas / Gulf Corridor (Lat 31.2, Lng -96.5)
2. **DBSCAN (Density-Based Spatial Clustering of Applications with Noise):**
   - Parameters: $\epsilon = 0.7$, $\text{min\_samples} = 25$.
   - Captures non-spherical highway corridors following actual roadways.
   - Identifies **18 dense collision corridors** while filtering **432 isolated noise incidents** ($\text{cluster} = -1$).
3. **PCA (Principal Component Analysis):**
   - Projects 49 features into orthogonal principal components.
   - Generates Scree plot variance ratios: Top 10 PCs explain **76.4% cumulative variance**.
4. **t-SNE (t-Distributed Stochastic Neighbor Embedding):**
   - Non-linear 2D manifold embedding mapping high-dimensional incident attributes into visual clusters.
5. **Isolation Forest (Anomaly Detection):**
   - Contamination rate: 3%.
   - Isolates **1,680 critical outlier hazards** (e.g., massive multi-car pileups during blizzard squalls).

---

### 3.5 `src/co5_evaluation_tuning.py`
#### What it is:
Performance benchmarking, probability calibration, and hyperparameter search engine.
#### Key Components:
1. **Stratified 5-Fold Cross-Validation:**
   - Evaluates generalization variance across 5 folds with preserved class proportions.
   - LightGBM: Mean accuracy $90.59\% \pm 0.33\%$, Macro F1: $0.9095 \pm 0.0031$.
2. **Comprehensive Classification Metrics:**
   - Multi-class Confusion Matrix (4x4).
   - Per-class Precision, Recall, F1-Score, and One-vs-Rest (OvR) ROC / PR curves.
3. **Probability Calibration:**
   - Computes the **Brier Score**: $\text{BS} = \frac{1}{N} \sum (f_{ik} - o_{ik})^2$.
   - Achieves a calibrated score of **0.0601**, ensuring predicted probabilities match historical occurrence rates.
4. **Hyperparameter Optimization Comparison:**
   - **Grid Search (`GridSearchCV`):** Exhaustive search across Cartesian product (8 candidate fits, 89.84% accuracy).
   - **Random Search (`RandomizedSearchCV`):** Samples combinations from distributions (6 candidates, 89.62% accuracy in 40% less compute time).
   - **Bayesian Optimization:** Balances exploration and exploitation via Expected Improvement (EI).

---

### 3.6 `src/co6_mlops_monitoring.py`
#### What it is:
MLOps production engineering, Feature Store management, and real-time distribution drift monitoring.
#### Key Classes:
- **`FeatureStore`**:
  - Central schema defined in `data/feature_store.json`.
  - Implements `enrich_and_validate()`: Supplies default median values for missing numerical sensors, fills categorical fallbacks, and derives features (`Rush_Hour`, `Is_Weekend`) during inference.
- **`DriftMonitor`**:
  - **Two-Sample Kolmogorov-Smirnov Test (KS-Test):** Compares live incoming continuous features against baseline data. Flags a drift alert when $p < 0.05$.
  - **Population Stability Index (PSI):** Computes $\text{PSI} = \sum (A_i - E_i) \ln(A_i / E_i)$ to verify categorical stability ($\text{PSI} < 0.1$).
  - Tracks rolling inference latency percentiles (**p50: 12.4 ms, p95: 28.1 ms, p99: 45.3 ms**).

---

### 3.7 `src/inference_pipeline.py`
#### What it is:
Production inference orchestrator combining the Feature Store, preprocessor, classifier, and regressor.
#### Key Methods:
- **`predict(raw_input)`**:
  - Enriches and validates payload.
  - Transforms features via `ColumnTransformer`.
  - Predicts severity class, calibrated probability distribution, expected clearance duration, and 0-100 Risk Index.
  - Generates recommended emergency response protocols.
- **`trace_prediction_lifecycle(raw_input)` (CO1 Mandatory Requirement):**
  - Produces a 6-stage provenance audit report tracing the request back through:
    1. *Data Ingestion & Hygiene* (provenance back to US Accidents 7.7M corpus).
    2. *Feature Engineering & Feature Store* (schema validation, imputation, derived features).
    3. *Model Training Lineage* (algorithm architecture, parameters, loss function).
    4. *Evaluation Verification Gate* (cross-validation score, Brier calibration check).
    5. *Deployment & REST Serving* (Flask endpoint, latency in ms).
    6. *Continuous Drift Monitoring* (KS-test and PSI verification).

---

### 3.8 `app.py` (REST API & Web Controller)
#### What it is:
Flask web microservice serving both the frontend user interface and REST API endpoints.
#### Endpoints:
- `GET /`: Renders the full interactive dashboard (`templates/index.html`).
- `POST /api/predict`: Real-time inference endpoint returning severity, probabilities, and recommendations.
- `POST /api/lifecycle_trace` & `GET /api/lifecycle_trace`: Returns the 6-stage lifecycle trace.
- `GET /api/metrics`: Returns `models/all_models_metrics.json` containing benchmark data across all COs.
- `GET /api/monitoring`: Runs live two-sample KS drift tests and returns system health telemetry.
- `GET /api/hotspots`: Returns K-Means and DBSCAN spatial cluster coordinates.
- `GET /api/health`: Health-check endpoint verifying model loading and server status.

---

### 3.9 Frontend Architecture
- **`templates/index.html`**:
  - Responsive single-page dashboard with 7 functional tabs:
    1. *Live Severity Predictor* (interactive form, presets, risk meter, probability bars, emergency advice).
    2. *Lifecycle Tracer (CO1)* (visual audit timeline for the prediction request).
    3. *Model Benchmarks (CO2 & CO3)* (tables for linear models, tree models, feature importances).
    4. *Hotspots & Clusters (CO4)* (K-Means regional zones, DBSCAN density corridors, PCA scree data).
    5. *Evaluation & Tuning (CO5)* (5-Fold CV metrics, calibration cards, hyperparameter search tables).
    6. *MLOps & Drift (CO6)* (live KS-test table, latency percentiles, drift simulation).
    7. *Course Outcomes Matrix* (complete syllabus mapping).
- **`static/css/style.css`**:
  - Clean styling with color-coded severity tiers, badges, and responsive CSS grid layouts.
- **`static/js/app.js`**:
  - Asynchronous AJAX logic for form submission, dynamic result rendering, and drift simulation.

---

### 3.10 Orchestration & Validation Scripts
- **`scripts/extract_sample.py`**: Reads raw `US_Accidents_March23.csv` in chunks and exports the balanced 80,000-record sample (`data/roadsafe_clean_sample.csv`).
- **`scripts/run_training.py`**: Executes the training workflow across all COs, trains all models, and serializes artifacts into `models/`.
- **`test_project.py`**: Automated test suite containing 6 unit and integration tests verifying all endpoints and models.

---

## 4. Course Outcomes (CO) Rigorous Verification Matrix

| Course Outcome | Syllabus Requirement | Implementation Details | Verified Metric / Output |
| :--- | :--- | :--- | :--- |
| **CO1** | End-to-end ML system lifecycle analysis and single-request lifecycle tracing. | Bidirectional provenance engine in `src/inference_pipeline.py`. Traces incoming HTTP request backwards through all 6 stages. | Endpoint: `POST /api/lifecycle_trace`<br>Audit latency: ~13.8 ms |
| **CO2** | Linear models (OLS, Ridge, Lasso, ElasticNet, Logistic, Multinomial). Reasoning on regularisation, bias-variance, and scaling. | Continuous duration regression and classification in `src/co2_linear_models.py`. Analyzed $L_1$ vs $L_2$ shrinkage and `StandardScaler` necessity. | Lasso: **49.0% sparsity (24 zero weights)**<br>Multinomial LogReg: 71.4% Acc |
| **CO3** | Tree-based models (DT, RF, XGBoost, LightGBM). Reasoning on feature importance, splitting criteria, ensemble variance reduction. | Gini vs Entropy comparison. Random Forest bagging variance reduction vs LightGBM gradient bias reduction in `src/co3_tree_models.py`. | **LightGBM: 91.8% Accuracy, 0.9880 ROC-AUC**<br>Top Feature: Distance(mi) |
| **CO4** | Unsupervised learning (K-Means, Hierarchical, DBSCAN, PCA, t-SNE, Isolation Forest). | K-Means spatial zones, DBSCAN corridor density, PCA scree variance, and Isolation Forest anomaly detection in `src/co4_unsupervised.py`. | 5 K-Means clusters<br>18 DBSCAN corridors<br>1,680 Anomaly hazards |
| **CO5** | Cross-validation, splits, classification/regression metrics, calibration, and hyperparameter search. | Stratified 70/15/15 splits, Stratified 5-Fold CV, ROC/PR curves, Brier score calibration, and Grid vs Random search in `src/co5_evaluation_tuning.py`. | 5-Fold CV: 90.59%<br>Brier Score: 0.0601<br>Tuning: Grid vs Random |
| **CO6** | MLOps engineering (Feature Store, skew avoidance, model packaging, REST serving, drift monitoring). | Centralized Feature Store catalog, serialized `ColumnTransformer`, Flask microservice, and two-sample KS-test drift monitor in `src/co6_mlops_monitoring.py`. | Real-time KS-test ($p$-values)<br>Latencies: p50 12.4ms, p95 28.1ms |

---

## 5. Mathematical & Theoretical Foundations

### 5.1 Regularization Formulations (CO2)
$$\text{Ridge: } \min_w \frac{1}{2n} \|y - Xw\|_2^2 + \alpha \|w\|_2^2 \quad \text{vs.} \quad \text{Lasso: } \min_w \frac{1}{2n} \|y - Xw\|_2^2 + \alpha \|w\|_1$$
- **Geometry of Sparsity:** The $L_1$ ball is a cross-polytope with sharp corners on the coordinate axes. The elliptical contours of the least-squares loss typically intersect the $L_1$ diamond at a corner, driving parameters to exactly $0.0$.
- **Bias-Variance Decomposition:**
  $$\mathbb{E}[(y - \hat{f}(x))^2] = \text{Bias}[\hat{f}(x)]^2 + \text{Var}[\hat{f}(x)] + \sigma^2$$
  Shrinkage parameter $\alpha$ intentionally adds a small amount of bias to significantly compress estimation variance.

### 5.2 Tree Splitting & Ensemble Mathematics (CO3)
- **Gini Impurity:** $\text{Gini}(D) = 1 - \sum_{i=1}^C p_i^2$
- **Shannon Entropy:** $\text{Entropy}(D) = -\sum_{i=1}^C p_i \log_2(p_i)$
- **Random Forest Bagging Variance Reduction:**
  $$\text{Var}(\bar{f}) = \rho \sigma^2 + \frac{1 - \rho}{B} \sigma^2$$
  Randomized feature subspace selection reduces pairwise correlation $\rho$, minimizing total ensemble variance.
- **Gradient Boosting Pseudo-Residuals:**
  $$r_{im} = -\left[\frac{\partial L(y_i, F(x_i))}{\partial F(x_i)}\right]_{F=F_{m-1}}$$
  Sequentially fits new decision trees to the negative gradient of the multi-class log-loss.

### 5.3 Unsupervised & Drift Formulations (CO4 & CO6)
- **K-Means Inertia:** $J = \sum_{k=1}^K \sum_{x \in S_k} \|x - \mu_k\|^2$
- **Silhouette Coefficient:** $s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$
- **Two-Sample Kolmogorov-Smirnov Test:**
  $$D = \sup_x |F_{\text{baseline}}(x) - F_{\text{live}}(x)|$$
- **Population Stability Index (PSI):**
  $$\text{PSI} = \sum_{i=1}^B (A_i - E_i) \times \ln\left(\frac{A_i}{E_i}\right)$$

---

## 6. Empirical Benchmark Results

### 6.1 Multi-Class Severity Classification (CO3 & CO5)
| Model | Test Accuracy | Macro F1 | Macro Precision | Macro Recall | ROC-AUC (OvR) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **LightGBM (Leaf-Wise)** | **91.77%** | **92.09%** | **92.41%** | **91.83%** | **0.9880** |
| **XGBoost (Extreme Gradient)** | 90.16% | 90.69% | 91.09% | 90.38% | 0.9849 |
| **Decision Tree (Gini)** | 83.57% | 84.57% | 85.82% | 84.12% | 0.9578 |
| **Random Forest (Bagging)** | 83.47% | 85.21% | 85.72% | 84.78% | 0.9664 |
| **Decision Tree (Entropy)** | 82.46% | 83.80% | 84.53% | 83.35% | 0.9577 |
| **Multinomial Logistic Regression** | 71.42% | 74.10% | 75.23% | 73.48% | 0.8842 |

### 6.2 Clearance Duration Regression (CO2)
| Model | Test $R^2$ | Test RMSE | Test MAE | Zero Weights (Sparsity) |
| :--- | :--- | :--- | :--- | :--- |
| **Linear Regression (OLS)** | 0.1703 | 55.44 min | 28.82 min | 2.0% (1 zero) |
| **Ridge Regression ($L_2$)** | 0.1702 | 55.44 min | 28.82 min | 2.0% (1 zero) |
| **Lasso Regression ($L_1$)** | 0.1659 | 55.58 min | 28.79 min | **49.0% (24 zeros)** |
| **ElasticNet (Hybrid)** | 0.1583 | 55.84 min | 28.81 min | 34.7% (17 zeros) |

---

## 7. How to Run and Present for Evaluation

### Quickstart Execution:
```powershell
# 1. Run the Web Application
& "C:\Users\pathi\OneDrive\Desktop\ML\.venv\Scripts\python.exe" "C:\Users\pathi\OneDrive\Desktop\ML\ClassProject\RoadSafe_AI\app.py"

# 2. Access the Dashboard
# Open browser at http://127.0.0.1:5000

# 3. Run Automated Tests
& "C:\Users\pathi\OneDrive\Desktop\ML\.venv\Scripts\python.exe" "C:\Users\pathi\OneDrive\Desktop\ML\ClassProject\RoadSafe_AI\test_project.py"
```

### Presentation Flow for Faculty Evaluation:
1. **Introduction:** State the problem and project title (**RoadSafe: AI-Based Road Accident Severity Prediction and Traffic Risk Analysis System**).
2. **CO1 Demonstration:** Go to the web dashboard, submit an accident prediction, and open the **Lifecycle Tracer (CO1)** tab to show the 6-stage provenance audit report.
3. **CO2 & CO3 Benchmarks:** Open the **Model Benchmarks** tab to present the comparison tables, explaining why Lasso induced 49% sparsity and how LightGBM achieved 91.8% accuracy and 0.9880 ROC-AUC.
4. **CO4 Hotspots:** Open the **Hotspots & Clusters** tab to show the 5 K-Means regional zones, 18 DBSCAN density corridors, and Isolation Forest anomaly detection.
5. **CO5 & CO6 Validation:** Highlight the Stratified 5-Fold CV metrics (90.59%), Brier score calibration (0.0601), Feature Store catalog, and live KS-test drift monitoring.
