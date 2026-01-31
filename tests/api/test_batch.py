"""
Tests for POST /predict/batch endpoint.
"""

class TestBatchPredictEndpoint:
    """Tests for batch prediction endpoint."""
    
    def test_batch_predict_returns_200_with_valid_texts(self, client):
        """Batch predict should return 200 with valid input."""
        response = client.post('/predict/batch', json={
            'texts': ['Article one.', 'Article two.']
        })
        assert response.status_code == 200
    
    def test_batch_predict_returns_list_of_predictions(self, client):
        """Batch predict should return predictions for each text."""
        texts = ['Article one.', 'Article two.', 'Article three.']
        response = client.post('/predict/batch', json={'texts': texts})
        data = response.get_json()
        assert 'predictions' in data
        assert len(data['predictions']) == len(texts)
    
    def test_batch_predict_each_prediction_has_required_fields(self, client):
        """Each prediction in batch should have all required fields."""
        response = client.post('/predict/batch', json={
            'texts': ['Test article.']
        })
        data = response.get_json()
        prediction = data['predictions'][0]
        assert 'credibility_score' in prediction
        assert 'label' in prediction
        assert 'probability' in prediction


class TestBatchPredictValidation:
    """Tests for input validation on /predict/batch endpoint."""
    
    def test_batch_predict_returns_400_without_texts(self, client):
        """Batch predict should return 400 if texts is missing."""
        response = client.post('/predict/batch', json={})
        assert response.status_code == 400
    
    def test_batch_predict_returns_400_with_empty_list(self, client):
        """Batch predict should return 400 if texts list is empty."""
        response = client.post('/predict/batch', json={'texts': []})
        assert response.status_code == 400
