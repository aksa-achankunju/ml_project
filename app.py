"""
RoadSafe AI - Production Flask Application & RESTful Inference Server
Provides web interface and REST API endpoints fulfilling CO1 (Lifecycle Tracer) and CO6 (Production Serving).
"""

import os
import sys
import json
import time
from flask import Flask, render_template, request, jsonify

# Add root directory to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.inference_pipeline import RoadSafePredictor
from src.co6_mlops_monitoring import DriftMonitor

app = Flask(
    __name__,
    template_folder=os.path.join(PROJECT_ROOT, 'templates'),
    static_folder=os.path.join(PROJECT_ROOT, 'static')
)

# Initialize predictor singleton
predictor = RoadSafePredictor()
drift_monitor = DriftMonitor()

@app.route('/')
def home():
    """Serves the main interactive dashboard."""
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """
    POST /api/predict: Real-time inference endpoint for accident severity prediction.
    """
    try:
        raw_data = request.get_json(force=True)
        if not raw_data:
            return jsonify({"error": "No input payload provided"}), 400

        result = predictor.predict(raw_data)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/lifecycle_trace', methods=['GET', 'POST'])
def api_lifecycle_trace():
    """
    CO1 End-to-End Prediction Lifecycle Trace endpoint.
    Traces one prediction request backwards through every preceding lifecycle stage.
    """
    try:
        if request.method == 'POST':
            payload = request.get_json(force=True)
        else:
            # Default payload
            payload = {
                "Temperature(F)": 62.5,
                "Humidity(%)": 78,
                "Pressure(in)": 29.85,
                "Visibility(mi)": 6.0,
                "Wind_Speed(mph)": 12.0,
                "Precipitation(in)": 0.15,
                "Weather_Condition": "Rain",
                "Sunrise_Sunset": "Night",
                "State_Clean": "PA",
                "Hour": 19,
                "Month": 11,
                "Start_Lat": 40.4406,
                "Start_Lng": -79.9959,
                "Junction": 1,
                "Traffic_Signal": 0
            }

        trace = predictor.trace_prediction_lifecycle(payload)
        return jsonify(trace)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/metrics', methods=['GET'])
def api_metrics():
    """
    Returns benchmark metrics across all course outcomes (CO2, CO3, CO4, CO5, CO6).
    """
    metrics_path = os.path.join(PROJECT_ROOT, 'models', 'all_models_metrics.json')
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            data = json.load(f)
        return jsonify(data)
    return jsonify({"error": "Metrics artifact not found. Please run training script first."}), 404

@app.route('/api/monitoring', methods=['GET'])
def api_monitoring():
    """
    Runs live drift detection tests (Two-sample KS-Test and PSI) and returns latency metrics.
    """
    drift_report = drift_monitor.check_drift()
    return jsonify(drift_report)

@app.route('/api/hotspots', methods=['GET'])
def api_hotspots():
    """
    Returns K-Means spatial cluster centroids and DBSCAN dense accident corridors (CO4).
    """
    metrics_path = os.path.join(PROJECT_ROOT, 'models', 'all_models_metrics.json')
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            data = json.load(f)
        unsupervised = data.get("co4_unsupervised", {})
        return jsonify(unsupervised)
    return jsonify({"error": "Hotspots data not found"}), 404

@app.route('/api/health', methods=['GET'])
def api_health():
    """Service health check endpoint."""
    return jsonify({
        "status": "HEALTHY",
        "service": "RoadSafe AI Inference Server",
        "models_loaded": {
            "preprocessor": predictor.preprocessor is not None,
            "classifier": predictor.clf_model is not None,
            "regressor": predictor.reg_model is not None
        },
        "timestamp": time.time()
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"\n========================================================")
    print(f" RoadSafe AI Server running at http://127.0.0.1:{port} ")
    print(f"========================================================\n")
    app.run(host='0.0.0.0', port=port, debug=False)
