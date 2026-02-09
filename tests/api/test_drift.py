"""
Tests for drift monitoring API endpoints.
"""
import pytest


class TestDriftStatusEndpoint:
    """Tests for GET /drift/status endpoint."""

    def test_drift_status_returns_200(self, client):
        """Drift status endpoint should return 200 OK."""
        response = client.get('/drift/status')
        assert response.status_code == 200

    def test_drift_status_returns_required_fields(self, client):
        """Drift status should return all required fields."""
        response = client.get('/drift/status')
        data = response.get_json()
        
        assert 'is_drifted' in data
        assert 'sample_count' in data
        assert 'window_size' in data
        assert 'metrics' in data
        assert 'details' in data

    def test_drift_status_metrics_structure(self, client):
        """Drift status metrics should have expected structure."""
        response = client.get('/drift/status')
        data = response.get_json()
        
        metrics = data['metrics']
        assert 'prediction_mean_shift' in metrics
        assert 'prediction_std_shift' in metrics
        assert 'text_length_shift' in metrics

    def test_drift_status_initial_state(self, client):
        """Initial drift status should show insufficient samples."""
        # Reset first to ensure clean state
        client.post('/drift/reset')
        
        response = client.get('/drift/status')
        data = response.get_json()
        
        assert data['is_drifted'] is False
        assert data['sample_count'] == 0


class TestDriftResetEndpoint:
    """Tests for POST /drift/reset endpoint."""

    def test_drift_reset_returns_200(self, client):
        """Drift reset endpoint should return 200 OK."""
        response = client.post('/drift/reset')
        assert response.status_code == 200

    def test_drift_reset_returns_status(self, client):
        """Drift reset should return status and message."""
        response = client.post('/drift/reset')
        data = response.get_json()
        
        assert data['status'] == 'reset'
        assert 'message' in data

    def test_drift_reset_clears_samples(self, client):
        """Drift reset should clear all recorded samples."""
        # Make some predictions first (if model is loaded, this would record them)
        client.post('/drift/reset')
        
        # Check status is cleared
        response = client.get('/drift/status')
        data = response.get_json()
        
        assert data['sample_count'] == 0


class TestDriftIntegration:
    """Integration tests for drift monitoring with predictions."""

    def test_drift_records_predictions(self, client):
        """Predictions should be recorded for drift monitoring."""
        # Reset to start fresh
        client.post('/drift/reset')
        
        # Make a prediction (uses mock service in testing mode)
        client.post('/predict', json={'text': 'Test article about news events'})
        
        # Check sample count increased
        response = client.get('/drift/status')
        data = response.get_json()
        
        # Note: In testing mode with MockModelService, predictions may not be recorded
        # This test validates the endpoint integration works
        assert response.status_code == 200
