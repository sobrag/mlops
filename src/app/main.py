"""
Flask API for Fake News Detection.

Endpoints:
- GET /health: Health check
- POST /predict: Single text prediction
- POST /predict/batch: Batch predictions
- GET /drift/status: Drift monitoring status
- POST /drift/reset: Reset drift monitoring window
- GET /metrics: Prometheus metrics
"""
from __future__ import annotations
import os
from flask import Flask
from flask_cors import CORS

from src.app.routes import health_bp, predict_bp, drift_bp, set_model_service, reset_model_service
from src.app.services import MockModelService, ModelService


def create_app(testing: bool = False) -> Flask:
    """
    Application factory.
    
    Args:
        testing: If True, uses mock model for testing
    """
    app = Flask(__name__)
    CORS(app)

    # Prometheus metrics (disabled in testing)
    if not testing:
        from prometheus_flask_exporter import PrometheusMetrics
        from prometheus_client import Gauge
        
        metrics = PrometheusMetrics(app)
        # Add app info
        metrics.info('app_info', 'Fake News Detection API', version='1.0.0')
        
        # Drift monitoring gauges
        drift_status_gauge = Gauge('drift_is_drifted', 'Whether drift has been detected (0 or 1)')
        drift_sample_count = Gauge('drift_sample_count', 'Number of predictions in drift window')
        drift_prob_shift = Gauge('drift_prediction_mean_shift', 'Shift in prediction mean from reference')
        drift_length_shift = Gauge('drift_text_length_shift', 'Shift in text length from reference')
        
        # Background task to update drift metrics
        def update_drift_metrics():
            from src.app.drift_monitor import get_drift_monitor
            try:
                monitor = get_drift_monitor()
                status = monitor.get_status()
                drift_status_gauge.set(1 if status['is_drifted'] else 0)
                drift_sample_count.set(status['sample_count'])
                drift_prob_shift.set(status['metrics']['prediction_mean_shift'])
                drift_length_shift.set(status['metrics']['text_length_shift'])
            except Exception:
                pass
        
        # Register callback to update metrics on each request
        @app.after_request
        def after_request_drift_update(response):
            update_drift_metrics()
            return response

    reset_model_service()
    
    if testing:
        set_model_service(MockModelService())
    else:
        service = ModelService()
        if not service.load():
            app.logger.warning("Model not loaded. Predictions will fail.")
        set_model_service(service)
    
    # Register blueprints
    app.register_blueprint(health_bp)
    app.register_blueprint(predict_bp)
    app.register_blueprint(drift_bp)
    
    return app


if __name__ == '__main__':
    app = create_app()
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
