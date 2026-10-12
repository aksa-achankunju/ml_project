# 🚦 RoadSafe AI: AI-Based Road Accident Severity Prediction & Traffic Risk Analysis System

An end-to-end Machine Learning system developed for academic evaluation, meeting **Course Outcomes CO1 through CO6**.

---

## 🌟 Key Features
- **Accident Severity Prediction (CO3):** Multi-class severity classification (Severity 1 to 4) using Calibrated LightGBM (91.8% Accuracy, 0.9880 ROC-AUC).
- **Clearance Duration Regression (CO2):** Linear, Ridge (L2), Lasso (L1, 49% sparsity), and ElasticNet models for impact delay estimation.
- **Single-Request Lifecycle Tracer (CO1):** Bidirectional traceability tracking an active HTTP prediction request back through all 6 ML lifecycle stages.
- **Hotspots & Spatial Clustering (CO4):** K-Means regional zones, DBSCAN dense highway collision corridors, PCA scree variance, and Isolation Forest anomaly detection.
- **Evaluation & Calibration (CO5):** Stratified 5-Fold Cross-Validation (90.6% CV accuracy), Brier score calibration (0.0601), and hyperparameter search analysis.
- **Production MLOps (CO6):** Reusable Feature Store, serialized ColumnTransformer pipeline, two-sample Kolmogorov-Smirnov drift monitoring, and REST microservice serving.
- **Interactive Web Interface:** Modern, responsive HTML5 + CSS3 dashboard with real-time prediction forms, risk meters, and telemetry graphs.

---

## 📁 Project Directory Structure
```
RoadSafe_AI/
│
├── data/
│   ├── roadsafe_clean_sample.csv       # Balanced 80,000 incident dataset
│   ├── reference_baseline.csv          # Baseline distribution for drift detection
│   └── feature_store.json              # Canonical Feature Store schema & statistics
│
├── models/
│   ├── accident_severity_model.joblib  # Calibrated LightGBM classifier
│   ├── accident_duration_model.joblib  # Ridge regression duration model
│   ├── preprocessor_pipeline.joblib    # Reusable Scikit-Learn ColumnTransformer
│   ├── kmeans_hotspots.joblib          # K-Means spatial clustering model
│   ├── isolation_forest.joblib         # Isolation Forest anomaly detector
│   ├── all_models_metrics.json         # Comprehensive metrics benchmark artifact
│   └── model_metadata.json             # System versioning & metadata
│
├── src/
│   ├── data_preprocessing.py           # Preprocessor & train/val/test split logic
│   ├── co2_linear_models.py            # Linear, Ridge, Lasso, ElasticNet, Logistic
│   ├── co3_tree_models.py              # Decision Trees, Random Forest, XGBoost, LightGBM
│   ├── co4_unsupervised.py             # K-Means, DBSCAN, PCA, t-SNE, Isolation Forest
│   ├── co5_evaluation_tuning.py        # 5-Fold CV, ROC/PR curves, Calibration, Tuning
│   ├── co6_mlops_monitoring.py         # Feature Store, drift detection (KS-test & PSI)
│   └── inference_pipeline.py           # Production inference & CO1 request tracing
│
├── static/
│   ├── css/style.css                   # Modern CSS3 styling (cards, badges, timelines)
│   └── js/app.js                       # Interactive AJAX logic, tabs, and form handling
│
├── templates/
│   └── index.html                      # Full dashboard web interface
│
├── notebooks/
│   └── RoadSafe_Comprehensive_Evaluation.ipynb # Step-by-step academic Jupyter Notebook
│
├── scripts/
│   ├── extract_sample.py               # Data extraction and preprocessing script
│   ├── run_training.py                 # Full automated training orchestrator
│   └── generate_notebook.py            # Notebook generation script
│
├── app.py                              # Flask REST API and web application server
├── test_project.py                     # Automated unit and integration test suite
├── requirements.txt                    # Python dependencies
├── PROJECT_REPORT.md                   # Full academic report with mathematical formulations
└── README.md                           # Quickstart guide
```

---

## 🚀 Quickstart Guide

### 1. Installation
Activate your Python environment and install the required dependencies:
```bash
pip install -r requirements.txt
```

### 2. Launch the Web Dashboard
Start the production Flask server:
```bash
python app.py
```
Open your web browser and navigate to:
```
http://127.0.0.1:5000
```

### 3. Run Automated Tests
Execute the verification test suite:
```bash
python test_project.py
```

### 4. Re-run Model Training (Optional)
To retrain and re-evaluate all models across CO1 through CO6:
```bash
python scripts/run_training.py
```

---

## 📡 REST API Reference

### `POST /api/predict`
Predicts accident severity and risk metrics for a given set of conditions.
**Request Body:**
```json
{
  "Temperature(F)": 24.0,
  "Humidity(%)": 88,
  "Visibility(mi)": 1.5,
  "Wind_Speed(mph)": 22.0,
  "Precipitation(in)": 0.35,
  "Weather_Condition": "Snow",
  "Sunrise_Sunset": "Night",
  "State_Clean": "PA",
  "Hour": 21,
  "Month": 12,
  "Start_Lat": 40.44,
  "Start_Lng": -79.99,
  "Junction": 1,
  "Traffic_Signal": 0
}
```
**Response:**
```json
{
  "predicted_severity": 3,
  "severity_label": "Severity 3: Multi-lane blockage, heavy gridlock risk, potential injuries.",
  "risk_index": 74.2,
  "risk_category": "High Hazard - Serious Incident",
  "predicted_duration_minutes": 82.5,
  "class_probabilities": {
    "Severity_1": 0.032,
    "Severity_2": 0.215,
    "Severity_3": 0.684,
    "Severity_4": 0.069
  },
  "recommended_action": "Dispatch emergency medical services (EMS) and fire department.",
  "latency_ms": 13.8
}
```

### `POST /api/lifecycle_trace`
Traces the input request backwards through all 6 ML lifecycle stages (CO1).

### `GET /api/metrics`
Retrieves benchmark metrics across CO2, CO3, CO4, CO5, and CO6.

### `GET /api/monitoring`
Executes live Kolmogorov-Smirnov drift tests and reports latency percentiles.

---

## 📊 Course Outcomes Compliance Matrix
| Course Outcome | Syllabus Requirement | Project Implementation |
| :--- | :--- | :--- |
| **CO1** | ML Lifecycle & Request Traceability | Traces requests back through Ingestion, Features, Training, Validation, Serving, and Telemetry in `src/inference_pipeline.py`. |
| **CO2** | Linear Models, Regularization & Scaling | Implemented OLS, Ridge ($L_2$), Lasso ($L_1$, 49% sparsity), ElasticNet, Binary & Multinomial Logistic Regression in `src/co2_linear_models.py`. |
| **CO3** | Tree Ensembles & Variance Reduction | Decision Trees (Gini vs Entropy), Random Forest bagging variance reduction, and LightGBM boosting (91.8% accuracy, 0.9880 ROC-AUC) in `src/co3_tree_models.py`. |
| **CO4** | Unsupervised Learning & Hotspots | K-Means spatial clusters, DBSCAN density corridors, PCA scree analysis, and Isolation Forest anomaly detection in `src/co4_unsupervised.py`. |
| **CO5** | Evaluation, Cross-Validation & Calibration | Stratified 5-Fold Cross-Validation, Brier score calibration (0.0601), and GridSearchCV vs RandomizedSearchCV in `src/co5_evaluation_tuning.py`. |
| **CO6** | MLOps Engineering & Drift Monitoring | Reusable Feature Store catalog, serialized ColumnTransformer, Flask REST API, and two-sample KS-test drift monitor in `src/co6_mlops_monitoring.py`. |
