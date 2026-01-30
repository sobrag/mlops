"""
Tests for POST /predict endpoint.
"""

class TestPredictEndpoint:
    """Tests for single text prediction endpoint."""
    
    def test_predict_returns_200_with_valid_text(self, client):
        """Predict endpoint should return 200 with valid input."""
        response = client.post('/predict', json={'text': 'This is a news article.'})
        assert response.status_code == 200
    
    def test_predict_returns_credibility_score(self, client):
        """Predict endpoint should return a credibility_score."""
        response = client.post('/predict', json={'text': 'This is a news article.'})
        data = response.get_json()
        assert 'credibility_score' in data
        assert isinstance(data['credibility_score'], (int, float))
    
    def test_predict_score_in_valid_range(self, client):
        """Credibility score should be between 0 and 100."""
        response = client.post('/predict', json={'text': 'This is a news article.'})
        data = response.get_json()
        assert 0 <= data['credibility_score'] <= 100
    
    def test_predict_returns_label(self, client):
        """Predict endpoint should return a label (real/fake)."""
        response = client.post('/predict', json={'text': 'This is a news article.'})
        data = response.get_json()
        assert 'label' in data
        assert data['label'] in ['real', 'fake']
    
    def test_predict_returns_probability(self, client):
        """Predict endpoint should return probability."""
        response = client.post('/predict', json={'text': 'This is a news article.'})
        data = response.get_json()
        assert 'probability' in data
        assert 0 <= data['probability'] <= 1


class TestPredictValidation:
    """Tests for input validation on /predict endpoint."""
    
    def test_predict_returns_400_without_text(self, client):
        """Predict endpoint should return 400 if text is missing."""
        response = client.post('/predict', json={})
        assert response.status_code == 400
    
    def test_predict_returns_400_with_empty_text(self, client):
        """Predict endpoint should return 400 if text is empty."""
        response = client.post('/predict', json={'text': ''})
        assert response.status_code == 400
    
    def test_predict_returns_400_with_non_json(self, client):
        """Predict endpoint should return 400 if content is not JSON."""
        response = client.post('/predict', data='not json')
        assert response.status_code == 400
    
    def test_predict_error_response_format(self, client):
        """Error responses should have error field."""
        response = client.post('/predict', json={})
        data = response.get_json()
        assert 'error' in data
