"""
API route blueprints.
"""
from flask import Blueprint, request, jsonify
from src.app.services import ModelService, MockModelService

_model_service = None

def get_model_service(testing: bool = False):
    """Get or create model service instance."""
    global _model_service
    
    if _model_service is None:
        _model_service = ModelService()
        if not testing:
            _model_service.load()
    
    return _model_service


def set_model_service(service):
    """Set model service (for testing)."""
    global _model_service
    _model_service = service


def reset_model_service():
    """Reset model service to None."""
    global _model_service
    _model_service = None

health_bp = Blueprint('health', __name__)


@health_bp.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok"})


predict_bp = Blueprint('predict', __name__)


@predict_bp.route('/predict', methods=['POST'])
def predict():
    """
    Single text prediction endpoint.
    
    Request: {"text": "article text"}
    Response: {"credibility_score": 75.5, "probability": 0.755, "label": "real"}
    """
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 400
    
    data = request.get_json()
    
    if 'text' not in data:
        return jsonify({"error": "Missing required field: text"}), 400
    
    text = data['text']
    if not text or not isinstance(text, str) or text.strip() == '':
        return jsonify({"error": "text must be a non-empty string"}), 400
    
    try:
        service = get_model_service()
        result = service.predict(text)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@predict_bp.route('/predict/batch', methods=['POST'])
def predict_batch():
    """
    Batch prediction endpoint.
    
    Request: {"texts": ["article 1", "article 2", ...]}
    Response: {"predictions": [{...}, {...}, ...]}
    """
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 400
    
    data = request.get_json()
    
    if 'texts' not in data:
        return jsonify({"error": "Missing required field: texts"}), 400
    
    texts = data['texts']
    if not isinstance(texts, list) or len(texts) == 0:
        return jsonify({"error": "texts must be a non-empty list"}), 400
    
    try:
        service = get_model_service()
        predictions = service.predict_batch(texts)
        return jsonify({"predictions": predictions})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
