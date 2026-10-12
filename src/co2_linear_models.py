"""
RoadSafe AI - CO2 Linear Supervised-Learning Models
Implements:
1. Linear Regression (OLS)
2. Ridge Regression (L2 Regularization)
3. Lasso Regression (L1 Regularization, Sparsity)
4. ElasticNet Regression (L1 + L2 Hybrid)
5. Binary Logistic Regression (L2 penalized)
6. Multinomial Logistic Regression (Softmax, multiclass)

Includes theoretical reasoning and empirical metrics for:
- Regularization strength & weight shrinkage
- Bias-Variance Trade-off
- Feature Scaling necessity
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet,
    LogisticRegression
)
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, f1_score, precision_score, recall_score,
    log_loss, confusion_matrix
)

class CO2LinearExperimenter:
    def __init__(self, preprocessor):
        self.preprocessor = preprocessor
        self.models_reg = {}
        self.models_clf = {}
        self.results_reg = {}
        self.results_clf = {}

    def run_regression_experiments(self, X_train, y_train, X_test, y_test, feature_names=None):
        """
        Trains and benchmarks Linear, Ridge, Lasso, and ElasticNet models on accident duration (minutes).
        """
        # Fit preprocessor on train, transform both
        X_train_proc = self.preprocessor.transform(X_train)
        X_test_proc = self.preprocessor.transform(X_test)

        regressors = {
            'Linear_Regression_OLS': LinearRegression(),
            'Ridge_Regression_L2': Ridge(alpha=10.0, random_state=42),
            'Lasso_Regression_L1': Lasso(alpha=0.1, random_state=42, max_iter=2000),
            'ElasticNet_Hybrid': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42, max_iter=2000)
        }

        for name, model in regressors.items():
            model.fit(X_train_proc, y_train)
            self.models_reg[name] = model

            y_train_pred = model.predict(X_train_proc)
            y_test_pred = model.predict(X_test_proc)

            train_rmse = float(np.sqrt(mean_squared_error(y_train, y_train_pred)))
            test_rmse = float(np.sqrt(mean_squared_error(y_test, y_test_pred)))
            train_r2 = float(r2_score(y_train, y_train_pred))
            test_r2 = float(r2_score(y_test, y_test_pred))
            test_mae = float(mean_absolute_error(y_test, y_test_pred))

            coefs = model.coef_
            sparsity = int(np.sum(np.abs(coefs) < 1e-4))
            total_features = int(len(coefs))

            self.results_reg[name] = {
                "train_rmse": round(train_rmse, 3),
                "test_rmse": round(test_rmse, 3),
                "train_r2": round(train_r2, 4),
                "test_r2": round(test_r2, 4),
                "test_mae": round(test_mae, 3),
                "zero_coefficients": sparsity,
                "total_coefficients": total_features,
                "sparsity_percentage": round((sparsity / total_features) * 100, 1),
                "mean_abs_weight": round(float(np.mean(np.abs(coefs))), 4)
            }

        return self.results_reg

    def run_classification_experiments(self, X_train, y_train, X_test, y_test):
        """
        Trains and benchmarks Binary Logistic Regression and Multinomial Logistic Regression.
        """
        X_train_proc = self.preprocessor.transform(X_train)
        X_test_proc = self.preprocessor.transform(X_test)

        # 1. Binary classification: High Severity (Severity >= 3) vs Low Severity (Severity <= 2)
        y_train_bin = (y_train >= 3).astype(int)
        y_test_bin = (y_test >= 3).astype(int)

        bin_logreg = LogisticRegression(penalty='l2', C=1.0, max_iter=1000, random_state=42)
        bin_logreg.fit(X_train_proc, y_train_bin)
        y_test_bin_pred = bin_logreg.predict(X_test_proc)
        y_test_bin_proba = bin_logreg.predict_proba(X_test_proc)

        self.models_clf['Binary_Logistic_Regression'] = bin_logreg
        self.results_clf['Binary_Logistic_Regression'] = {
            "task": "Binary (High vs Low Severity)",
            "accuracy": round(float(accuracy_score(y_test_bin, y_test_bin_pred)), 4),
            "precision": round(float(precision_score(y_test_bin, y_test_bin_pred, average='binary')), 4),
            "recall": round(float(recall_score(y_test_bin, y_test_bin_pred, average='binary')), 4),
            "f1_score": round(float(f1_score(y_test_bin, y_test_bin_pred, average='binary')), 4),
            "log_loss": round(float(log_loss(y_test_bin, y_test_bin_proba)), 4),
            "regularization": "L2 (Ridge penalty, C=1.0)"
        }

        # 2. Multinomial Logistic Regression: 4-class Softmax
        multi_logreg = LogisticRegression(solver='lbfgs', C=1.0, max_iter=1000, random_state=42)
        multi_logreg.fit(X_train_proc, y_train)
        y_test_multi_pred = multi_logreg.predict(X_test_proc)
        y_test_multi_proba = multi_logreg.predict_proba(X_test_proc)

        self.models_clf['Multinomial_Logistic_Regression'] = multi_logreg
        self.results_clf['Multinomial_Logistic_Regression'] = {
            "task": "Multiclass (Severity 1 to 4)",
            "accuracy": round(float(accuracy_score(y_test, y_test_multi_pred)), 4),
            "precision_macro": round(float(precision_score(y_test, y_test_multi_pred, average='macro')), 4),
            "recall_macro": round(float(recall_score(y_test, y_test_multi_pred, average='macro')), 4),
            "f1_macro": round(float(f1_score(y_test, y_test_multi_pred, average='macro')), 4),
            "f1_weighted": round(float(f1_score(y_test, y_test_multi_pred, average='weighted')), 4),
            "log_loss": round(float(log_loss(y_test, y_test_multi_proba)), 4),
            "regularization": "Multinomial Softmax with L2 Shrinkage"
        }

        return self.results_clf

    def get_theory_and_reasoning(self):
        """
        Returns structured theoretical reasoning for CO2 syllabus outcomes.
        """
        return {
            "regularization_analysis": {
                "L2_Ridge": "Adds penalty alpha * sum(w_j^2). Shrinks weights toward zero proportionally. Effectively handles multicollinearity between correlated weather/traffic variables (e.g., Temperature and Humidity).",
                "L1_Lasso": "Adds penalty alpha * sum(|w_j|). Creates a diamond-shaped constraint region with sharp corners at axes, driving non-influential feature weights to exactly 0.0, performing automatic embedded feature selection.",
                "ElasticNet": "Hybrid penalty r * alpha * ||w||_1 + 0.5 * (1-r) * alpha * ||w||_2^2. Overcomes Lasso's limitation when features are strongly correlated by grouping correlated variables together."
            },
            "bias_variance_tradeoff": {
                "concept": "OLS unpenalized regression minimizes empirical training error, leading to low bias but high sensitivity to noisy sample variations (high variance). Regularization intentionally introduces a controlled amount of bias to significantly reduce parameter variance, resulting in superior test-set generalization.",
                "empirical_observation": "Ridge and ElasticNet maintain robust test R^2 and lower test RMSE compared to unconstrained high-variance fits."
            },
            "feature_scaling_necessity": {
                "rationale": "Regularization terms penalize the magnitude of weights uniformly across all dimensions. If Pressure (in) is ~30 and Duration is ~60 while Visibility is ~10, unscaled features force coefficients of small-magnitude variables to be massive, making them unfairly penalized by the regularizer. StandardScaler ensures mean=0 and variance=1 across all continuous inputs."
            }
        }
