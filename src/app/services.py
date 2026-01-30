"""
Model service for loading and inference.
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional
from src.features.vectorize import TextVectorizer
from src.models.predict import predict_all
from src.utils.io import load_joblib

DEFAULT_ARTIFACTS_PATH = Path(__file__).parent.parent.parent / "artifacts"

class ModelService:
    """Service class to handle model loading and predictions."""
    
    def __init__(self, artifacts_path: Optional[Path] = None):
        self.artifacts_path = artifacts_path or DEFAULT_ARTIFACTS_PATH
        self.model = None
        self.vectorizer = None
        self._loaded = False
    
    def load(self) -> bool:
        """Load model and vectorizer from artifacts."""
        try:
            artifact_dirs = sorted(
                [d for d in self.artifacts_path.iterdir() if d.is_dir()],
                reverse=True
            )
            
            if not artifact_dirs:
                return False
            
            latest_dir = artifact_dirs[0]
            model_path = latest_dir / "model.joblib"
            vectorizer_path = latest_dir / "vectorizer.joblib"
            
            if not model_path.exists() or not vectorizer_path.exists():
                return False
            
            self.model = load_joblib(model_path)
            self.vectorizer = TextVectorizer.load(vectorizer_path)
            self._loaded = True
            return True
        except Exception:
            return False
    
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded
    
    def predict(self, text: str) -> dict:
        """Make prediction for a single text."""
        X = self.vectorizer.transform([text])
        proba, score, label = predict_all(self.model, X)
        
        return {
            "credibility_score": float(score[0]),
            "probability": float(proba[0]),
            "label": "real" if label[0] == 1 else "fake"
        }
    
    def predict_batch(self, texts: list[str]) -> list[dict]:
        """Make predictions for multiple texts."""
        X = self.vectorizer.transform(texts)
        proba, scores, labels = predict_all(self.model, X)
        
        return [
            {
                "credibility_score": float(scores[i]),
                "probability": float(proba[i]),
                "label": "real" if labels[i] == 1 else "fake"
            }
            for i in range(len(texts))
        ]


class MockModelService:
    """Mock model service for testing."""
    
    def is_loaded(self) -> bool:
        return True
    
    def predict(self, text: str) -> dict:
        return {
            "credibility_score": 75.0,
            "probability": 0.75,
            "label": "real"
        }
    
    def predict_batch(self, texts: list[str]) -> list[dict]:
        return [self.predict(text) for text in texts]
