"""
RoadSafe AI - Automated Verification & Test Suite
Verifies end-to-end functionality across CO1 to CO6, REST endpoints, and inference integrity.
"""

import os
import sys
import unittest
import json

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app import app
from src.inference_pipeline import RoadSafePredictor
from src.co6_mlops_monitoring import FeatureStore, DriftMonitor

class TestRoadSafeAI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = app.test_client()
        cls.predictor = RoadSafePredictor()

    def test_health_endpoint(self):
        response = self.client.get('/api/health')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data['status'], 'HEALTHY')
        self.assertTrue(data['models_loaded']['classifier'])
        self.assertTrue(data['models_loaded']['preprocessor'])

    def test_metrics_endpoint(self):
        response = self.client.get('/api/metrics')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn('co1_ml_lifecycle', data)
        self.assertIn('co2_linear_models', data)
        self.assertIn('co3_tree_models', data)
        self.assertIn('co4_unsupervised', data)
        self.assertIn('co5_evaluation_tuning', data)
        self.assertIn('co6_mlops', data)

    def test_predict_api(self):
        payload = {
            "Temperature(F)": 28.0,
            "Humidity(%)": 90,
            "Pressure(in)": 29.5,
            "Visibility(mi)": 2.0,
            "Wind_Speed(mph)": 25.0,
            "Precipitation(in)": 0.4,
            "Weather_Condition": "Snow",
            "Sunrise_Sunset": "Night",
            "State_Clean": "PA",
            "Hour": 22,
            "Month": 1,
            "Start_Lat": 40.44,
            "Start_Lng": -79.99,
            "Junction": 1,
            "Traffic_Signal": 0
        }
        response = self.client.post('/api/predict', json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn('predicted_severity', data)
        self.assertIn(data['predicted_severity'], [1, 2, 3, 4])
        self.assertIn('risk_index', data)
        self.assertIn('class_probabilities', data)
        self.assertIn('predicted_duration_minutes', data)

    def test_lifecycle_trace_co1(self):
        """Validates CO1 requirement: Tracing request backwards through all 6 stages."""
        payload = {
            "Temperature(F)": 72.0,
            "Humidity(%)": 50,
            "Visibility(mi)": 10.0,
            "State_Clean": "CA"
        }
        response = self.client.post('/api/lifecycle_trace', json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn('lifecycle_stages', data)
        stages = data['lifecycle_stages']
        self.assertEqual(len(stages), 6)
        self.assertEqual(stages[0]['stage_name'], "Data Ingestion & Provenance")
        self.assertEqual(stages[1]['stage_name'], "Feature Engineering & Feature Store")
        self.assertEqual(stages[2]['stage_name'], "Model Architecture & Training Lineage")
        self.assertEqual(stages[3]['stage_name'], "Model Evaluation & Verification Gate")
        self.assertEqual(stages[4]['stage_name'], "Deployment & REST Serving")
        self.assertEqual(stages[5]['stage_name'], "Telemetry, Drift & Continuous Monitoring")

    def test_drift_monitoring_co6(self):
        response = self.client.get('/api/monitoring')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn('status', data)
        self.assertIn('features_monitored', data)

    def test_frontend_home(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn("RoadSafe AI", html)
        self.assertIn("Live Severity Predictor", html)
        self.assertIn("Lifecycle Tracer (CO1)", html)

if __name__ == '__main__':
    unittest.main()
