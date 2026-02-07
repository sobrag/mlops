"""
Shared fixtures for Streamlit UI tests.
"""
import pytest


@pytest.fixture
def mock_api_success_response():
    """Mock successful API response."""
    return {
        "credibility_score": 82.3,
        "probability": 0.823,
        "label": "real",
    }


@pytest.fixture
def mock_api_error_response():
    """Mock API error response."""
    return {"error": "text must be a non-empty string"}


@pytest.fixture
def sample_text():
    """Sample news article text for testing."""
    return "Scientists discover new species of deep-sea fish in the Pacific Ocean."


@pytest.fixture
def api_url():
    """Default API URL for testing."""
    return "http://localhost:5001"
