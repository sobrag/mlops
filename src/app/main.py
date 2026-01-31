"""
Flask API for Fake News Detection.

Endpoints:
- GET /health: Health check
- POST /predict: Single text prediction
- POST /predict/batch: Batch predictions
- GET /metrics: Prometheus metrics
"""
from __future__ import annotations
import os
from flask import Flask
from flask_cors import CORS

from src.app.routes import health_bp, predict_bp, set_model_service, reset_model_service
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
        metrics = PrometheusMetrics(app)
        # Add app info
        metrics.info('app_info', 'Fake News Detection API', version='1.0.0')

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
    
    return app


if __name__ == '__main__':
    app = create_app()
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
