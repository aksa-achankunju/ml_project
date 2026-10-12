"""
RoadSafe AI - Comprehensive Model Training Orchestrator
Executes training pipelines across all Course Outcomes (CO1 - CO6),
evaluates metrics, serializes production models, and outputs all_models_metrics.json.
"""

import os
import sys
import json
import time
import joblib
import numpy as np
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_preprocessing import (
    load_data, get_train_val_test_data, create_preprocessor,
    ALL_FEATURE_COLS, NUMERICAL_FEATURES, CATEGORICAL_FEATURES, BINARY_FEATURES
)
from src.co2_linear_models import CO2LinearExperimenter
from src.co3_tree_models import CO3TreeExperimenter
from src.co4_unsupervised import CO4UnsupervisedExperimenter
from src.co5_evaluation_tuning import CO5EvaluationEngine
from src.co6_mlops_monitoring import FeatureStore, DriftMonitor

def train_and_evaluate_all():
    print("=" * 70)
    print(" RoadSafe AI: Comprehensive ML Lifecycle Training Orchestrator ")
    print("=" * 70)
    start_total_time = time.time()

    data_path = os.path.join(PROJECT_ROOT, 'data', 'roadsafe_clean_sample.csv')
    models_dir = os.path.join(PROJECT_ROOT, 'models')
    os.makedirs(models_dir, exist_ok=True)

    print(f"\n[Stage 1: Data & Features] Loading sample from {data_path}...")
    df = load_data(data_path)
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns.")
    print(f"Target distribution:\n{df['Severity'].value_counts().to_dict()}")

    # Stratified Train/Val/Test Split (CO5)
    print("\n[Stage 2: Stratified Splits] Creating 70/15/15 train/val/test partitions...")
    (X_train, X_val, X_test, 
     y_clf_train, y_clf_val, y_clf_test, 
     y_reg_train, y_reg_val, y_reg_test) = get_train_val_test_data(df)

    print(f"Train set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    print(f"Test set: {len(X_test)} samples")

    # Build and Fit Preprocessor (CO1 & CO6 Training-Serving Skew Prevention)
    print("\nFitting ColumnTransformer pipeline...")
    preprocessor = create_preprocessor()
    preprocessor.fit(X_train)

    # Get feature names after one-hot encoding
    try:
        cat_encoder = preprocessor.named_transformers_['cat'].named_steps['encoder']
        cat_feature_names = list(cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES))
    except Exception:
        cat_feature_names = [f"cat_{i}" for i in range(15)]
    all_transformed_feature_names = NUMERICAL_FEATURES + cat_feature_names + BINARY_FEATURES
    print(f"Total transformed features: {len(all_transformed_feature_names)}")

    # -------------------------------------------------------------
    # CO2: Linear Supervised-Learning Models
    # -------------------------------------------------------------
    print("\n" + "=" * 50)
    print(" [CO2] Evaluating Linear Supervised-Learning Models...")
    print("=" * 50)
    co2_exp = CO2LinearExperimenter(preprocessor)
    co2_reg_results = co2_exp.run_regression_experiments(X_train, y_reg_train, X_test, y_reg_test)
    co2_clf_results = co2_exp.run_classification_experiments(X_train, y_clf_train, X_test, y_clf_test)
    co2_theory = co2_exp.get_theory_and_reasoning()
    print("CO2 Regression Benchmark:", co2_reg_results)
    print("CO2 Classification Benchmark:", co2_clf_results)

    # -------------------------------------------------------------
    # CO3: Tree-Based Supervised-Learning Models
    # -------------------------------------------------------------
    print("\n" + "=" * 50)
    print(" [CO3] Evaluating Tree-Based Models (DT, RF, XGB, LightGBM)...")
    print("=" * 50)
    co3_exp = CO3TreeExperimenter(preprocessor)
    co3_results = co3_exp.run_all_tree_models(X_train, y_clf_train, X_test, y_clf_test, feature_names=all_transformed_feature_names)
    co3_theory = co3_exp.get_theory_and_reasoning()
    print("CO3 Tree Models Benchmark:", co3_results)

    # -------------------------------------------------------------
    # CO4: Unsupervised-Learning Techniques
    # -------------------------------------------------------------
    print("\n" + "=" * 50)
    print(" [CO4] Evaluating Unsupervised Learning (K-Means, DBSCAN, PCA, Isolation Forest)...")
    print("=" * 50)
    co4_exp = CO4UnsupervisedExperimenter(preprocessor)
    kmeans_results = co4_exp.run_kmeans_analysis(df)
    dbscan_results = co4_exp.run_dbscan_analysis(df)
    dim_reduction_results = co4_exp.run_pca_and_tsne(X_train)
    anomaly_results = co4_exp.run_anomaly_detection(X_train)
    print("K-Means Clusters:", len(kmeans_results['clusters']))
    print("DBSCAN Corridors:", dbscan_results['total_corridors_found'])
    print("PCA Scree Components:", len(dim_reduction_results['pca_explained_variance']))
    print("Anomaly Detection Status:", anomaly_results)

    # -------------------------------------------------------------
    # CO5: Model Evaluation, CV, Calibration & Tuning
    # -------------------------------------------------------------
    print("\n" + "=" * 50)
    print(" [CO5] Evaluating Performance, Cross-Validation, Calibration & Tuning...")
    print("=" * 50)
    co5_engine = CO5EvaluationEngine(preprocessor)
    best_tree_model = co3_exp.models.get('LightGBM')
    cv_results = co5_engine.run_stratified_kfold_cv(X_train, y_clf_train, n_splits=5)
    clf_eval_results = co5_engine.compute_full_classification_metrics(best_tree_model, X_test, y_clf_test)
    calib_results = co5_engine.analyze_calibration(best_tree_model, X_test, y_clf_test)
    hyperparam_results = co5_engine.compare_hyperparameter_searches(X_train, y_clf_train)
    print("5-Fold CV Results:", cv_results)
    print("Overall Test Accuracy:", clf_eval_results['overall_accuracy'])
    print("Calibration Brier Score:", calib_results['uncalibrated_brier_score'])

    # -------------------------------------------------------------
    # CO6: MLOps Packaging & Serialization
    # -------------------------------------------------------------
    print("\n" + "=" * 50)
    print(" [CO6] Packaging Production Models & Metadata Artifacts...")
    print("=" * 50)
    # Serialize Preprocessor
    joblib.dump(preprocessor, os.path.join(models_dir, 'preprocessor_pipeline.joblib'))
    # Serialize Best Severity Classifier (LightGBM)
    joblib.dump(best_tree_model, os.path.join(models_dir, 'accident_severity_model.joblib'))
    # Serialize Best Duration Regressor (Ridge)
    best_reg_model = co2_exp.models_reg.get('Ridge_Regression_L2')
    joblib.dump(best_reg_model, os.path.join(models_dir, 'accident_duration_model.joblib'))
    # Serialize K-Means & Isolation Forest
    if co4_exp.kmeans_model is not None:
        joblib.dump(co4_exp.kmeans_model, os.path.join(models_dir, 'kmeans_hotspots.joblib'))
    if co4_exp.isolation_forest_model is not None:
        joblib.dump(co4_exp.isolation_forest_model, os.path.join(models_dir, 'isolation_forest.joblib'))

    # Compile Comprehensive Metrics Artifact
    all_metrics = {
        "metadata": {
            "system_name": "RoadSafe AI",
            "version": "1.0.0",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "total_records_trained": len(X_train),
            "total_records_tested": len(X_test),
            "primary_classifier": "LightGBM Classifier",
            "primary_regressor": "Ridge Regression (L2)"
        },
        "co1_ml_lifecycle": {
            "stages": [
                "1. Data Ingestion & Hygiene (US Accidents 7.7M Corpus)",
                "2. Feature Engineering & Feature Store Catalog",
                "3. Multi-Algorithm Model Training (Linear, Ensembles, Unsupervised)",
                "4. Evaluation Gates & Calibration (Stratified 5-Fold CV, PR-AUC, Brier Score)",
                "5. REST API Serving (Flask Microservice, Training-Serving Skew Prevention)",
                "6. Continuous Monitoring (KS-Test Drift, PSI, Telemetry Tracker)"
            ]
        },
        "co2_linear_models": {
            "regression_comparison": co2_reg_results,
            "classification_comparison": co2_clf_results,
            "theoretical_reasoning": co2_theory
        },
        "co3_tree_models": {
            "comparison": co3_results,
            "feature_importances": co3_exp.feature_importances.get('LightGBM', []),
            "theoretical_reasoning": co3_theory
        },
        "co4_unsupervised": {
            "kmeans": kmeans_results,
            "dbscan": dbscan_results,
            "dimensionality_reduction": dim_reduction_results,
            "anomaly_detection": anomaly_results
        },
        "co5_evaluation_tuning": {
            "cross_validation": cv_results,
            "classification_metrics": clf_eval_results,
            "calibration": calib_results,
            "hyperparameter_searches": hyperparam_results
        },
        "co6_mlops": {
            "drift_monitoring": DriftMonitor().check_drift(),
            "pipeline_integrity": "Serialized unified ColumnTransformer with zero-leakage guarantee.",
            "serving_framework": "RESTful Flask Application (/api/predict, /api/lifecycle_trace)"
        }
    }

    metrics_path = os.path.join(models_dir, 'all_models_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nSaved comprehensive metrics JSON to {metrics_path}")

    metadata_path = os.path.join(models_dir, 'model_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(all_metrics['metadata'], f, indent=2)
    print(f"Saved model metadata to {metadata_path}")

    total_elapsed = round(time.time() - start_total_time, 2)
    print(f"\n[SUCCESS] Entire ML pipeline trained, evaluated, and packaged in {total_elapsed} seconds!")

if __name__ == '__main__':
    train_and_evaluate_all()
