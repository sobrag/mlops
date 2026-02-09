"""
Model service for loading and inference.
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import Optional
from src.models.predict import predict_all
from src.utils.artifacts import resolve_model_dir, load_model_bundle

DEFAULT_ARTIFACTS_PATH = Path(__file__).parent.parent.parent / "artifacts"

class ModelService:
    """Service class to handle model loading and predictions."""
    
    def __init__(self, artifacts_path: Optional[Path] = None):
        self.artifacts_path = artifacts_path or DEFAULT_ARTIFACTS_PATH
        self.model = None
        self.vectorizer = None
        self._loaded = False
    
    def load(self) -> bool:
        """Load model and vectorizer; prefer local cache, fallback to W&B artifact if configured."""
        try:
            model_dir = resolve_model_dir(
                artifacts_dir=self.artifacts_path,
                model_artifact=os.getenv("MODEL_ARTIFACT"),
                use_wandb=os.getenv("USE_WANDB", "false").lower() == "true",
                wandb_mode=os.getenv("WANDB_MODE", "online"),
                project=os.getenv("WANDB_PROJECT") or "mlops",
                entity=os.getenv("WANDB_ENTITY"),
            )
            self.vectorizer, self.model = load_model_bundle(model_dir)
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
        
        result = {
            "credibility_score": float(score[0]),
            "probability": float(proba[0]),
            "label": "real" if label[0] == 0 else "fake"
        }
        
        # Record for drift monitoring
        try:
            from src.app.drift_monitor import get_drift_monitor
            monitor = get_drift_monitor()
            monitor.record_prediction(
                probability=result["probability"],
                text=text,
                label=result["label"],
            )
        except Exception:
            pass  # Don't fail prediction if drift monitoring fails
        
        return result
    
    def predict_batch(self, texts: list[str]) -> list[dict]:
        """Make predictions for multiple texts."""
        X = self.vectorizer.transform(texts)
        proba, scores, labels = predict_all(self.model, X)
        
        results = [
            {
                "credibility_score": float(scores[i]),
                "probability": float(proba[i]),
                "label": "real" if labels[i] == 0 else "fake"
            }
            for i in range(len(texts))
        ]
        
        # Record for drift monitoring
        try:
            from src.app.drift_monitor import get_drift_monitor
            monitor = get_drift_monitor()
            for i, text in enumerate(texts):
                monitor.record_prediction(
                    probability=results[i]["probability"],
                    text=text,
                    label=results[i]["label"],
                )
        except Exception:
            pass  # Don't fail prediction if drift monitoring fails
        
        return results


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
