"""
Shared fixtures for API tests.
"""
import pytest

@pytest.fixture
def app():
    """Create application for testing."""
    from src.app.main import create_app
    app = create_app(testing=True)
    app.config['TESTING'] = True
    return app


@pytest.fixture
def client(app):
    """Create test client."""
    return app.test_client()
