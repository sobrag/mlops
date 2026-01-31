"""
Flask API application for Fake News Detection.
"""
from src.app.main import create_app
from src.app.services import ModelService, MockModelService

__all__ = ["create_app", "ModelService", "MockModelService"]
