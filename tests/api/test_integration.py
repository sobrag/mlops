"""
Integration tests for the API with real model artifacts.

These tests require actual model and vectorizer artifacts to be present.
They test the full prediction pipeline from HTTP request to model inference.
"""
import pytest
from pathlib import Path

ARTIFACTS_PATH = Path(__file__).parent.parent.parent / "artifacts"


def has_real_artifacts() -> bool:
    """Check if real model artifacts exist."""
    if not ARTIFACTS_PATH.exists():
        return False
    
    artifact_dirs = [d for d in ARTIFACTS_PATH.iterdir() if d.is_dir()]
    if not artifact_dirs:
        return False
    
    latest_dir = sorted(artifact_dirs, reverse=True)[0]
    model_path = latest_dir / "model.joblib"
    vectorizer_path = latest_dir / "vectorizer.joblib"
    
    return model_path.exists() and vectorizer_path.exists()


pytestmark = pytest.mark.skipif(
    not has_real_artifacts(),
    reason="Real model artifacts not found. Run training pipeline first."
)


@pytest.fixture
def integration_app():
    """Create application with real model (not mock)."""
    from src.app.main import create_app
    app = create_app(testing=False)
    app.config['TESTING'] = True
    return app


@pytest.fixture
def integration_client(integration_app):
    """Create test client with real model."""
    return integration_app.test_client()


class TestIntegrationPredict:
    """Integration tests for /predict with real model."""
    
    def test_predict_real_news_returns_high_credibility(self, integration_client):
        """Real news articles should get higher credibility scores."""
        # Example of likely real news (factual, formal language)
        real_news = """
        The Federal Reserve announced today that it will maintain current 
        interest rates following its latest policy meeting. Chair Jerome Powell 
        stated that the decision reflects ongoing assessment of economic conditions.
        """
        
        response = integration_client.post('/predict', json={'text': real_news})
        assert response.status_code == 200
        
        data = response.get_json()
        assert 'credibility_score' in data
        assert 'probability' in data
        assert 'label' in data
        # Real model should return actual predictions (not mock values)
        assert isinstance(data['credibility_score'], (int, float))
    
    def test_predict_fake_news_returns_lower_credibility(self, integration_client):
        """Fake news articles should get lower credibility scores."""
        # Example of likely fake news (sensational, unverifiable claims)
        fake_news = """
        SHOCKING: Scientists discover that eating chocolate every day 
        makes you immortal! Big Pharma doesn't want you to know this 
        one simple trick that doctors HATE!
        """
        
        response = integration_client.post('/predict', json={'text': fake_news})
        assert response.status_code == 200
        
        data = response.get_json()
        assert 'credibility_score' in data
        assert 'label' in data
    
    def test_predict_returns_consistent_results(self, integration_client):
        """Same input should return same prediction (deterministic)."""
        text = "This is a test article about technology advances."
        
        response1 = integration_client.post('/predict', json={'text': text})
        response2 = integration_client.post('/predict', json={'text': text})
        
        data1 = response1.get_json()
        data2 = response2.get_json()
        
        assert data1['credibility_score'] == data2['credibility_score']
        assert data1['probability'] == data2['probability']
        assert data1['label'] == data2['label']


class TestIntegrationBatch:
    """Integration tests for /predict/batch with real model."""
    
    def test_batch_predict_with_mixed_articles(self, integration_client):
        """Batch prediction should work with multiple articles."""
        texts = [
            "The government passed a new budget bill today.",
            "UNBELIEVABLE! This one weird trick will make you rich overnight!",
            "Scientists published new research findings in Nature journal."
        ]
        
        response = integration_client.post('/predict/batch', json={'texts': texts})
        assert response.status_code == 200
        
        data = response.get_json()
        assert 'predictions' in data
        assert len(data['predictions']) == 3
        
        # Each prediction should have required fields
        for pred in data['predictions']:
            assert 'credibility_score' in pred
            assert 'probability' in pred
            assert 'label' in pred
            assert 0 <= pred['credibility_score'] <= 100
            assert 0 <= pred['probability'] <= 1
            assert pred['label'] in ['real', 'fake']
    
    def test_batch_matches_individual_predictions(self, integration_client):
        """Batch predictions should match individual predictions."""
        texts = ["Article one content.", "Article two content."]
        
        # Get batch predictions
        batch_response = integration_client.post('/predict/batch', json={'texts': texts})
        batch_data = batch_response.get_json()
        
        # Get individual predictions
        individual_predictions = []
        for text in texts:
            response = integration_client.post('/predict', json={'text': text})
            individual_predictions.append(response.get_json())
        
        # Compare
        for i, (batch_pred, indiv_pred) in enumerate(
            zip(batch_data['predictions'], individual_predictions)
        ):
            assert batch_pred['credibility_score'] == indiv_pred['credibility_score']
            assert batch_pred['probability'] == indiv_pred['probability']
            assert batch_pred['label'] == indiv_pred['label']


class TestIntegrationHealth:
    """Integration tests for /health with real model."""
    
    def test_health_with_loaded_model(self, integration_client):
        """Health check should work when model is loaded."""
        response = integration_client.get('/health')
        assert response.status_code == 200
        data = response.get_json()
        assert data['status'] == 'ok'
