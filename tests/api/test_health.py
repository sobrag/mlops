"""
Tests for GET /health endpoint.
"""

class TestHealthEndpoint:
    """Tests for health check endpoint."""
    
    def test_health_returns_200(self, client):
        """Health endpoint should return 200 OK."""
        response = client.get('/health')
        assert response.status_code == 200
    
    def test_health_returns_status_ok(self, client):
        """Health endpoint should return status: ok."""
        response = client.get('/health')
        data = response.get_json()
        assert data['status'] == 'ok'
