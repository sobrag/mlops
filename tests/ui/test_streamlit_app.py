"""
Tests for Streamlit UI — call_predict_api function.
"""
from unittest.mock import patch, MagicMock

import requests


class TestCallPredictApi:
    """Tests for the API call logic."""

    def test_successful_prediction(
        self, sample_text, api_url, mock_api_success_response
    ):
        """call_predict_api returns parsed result on 200 OK."""
        from src.ui.app import call_predict_api

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = mock_api_success_response

        with patch("src.ui.app.requests.post", return_value=mock_resp) as mock_post:
            result = call_predict_api(sample_text, api_url)

        mock_post.assert_called_once_with(
            f"{api_url}/predict",
            json={"text": sample_text},
            timeout=30,
        )
        assert result == mock_api_success_response

    def test_connection_error_returns_error_dict(self, sample_text, api_url):
        """call_predict_api returns error dict when API is unreachable."""
        from src.ui.app import call_predict_api

        with patch(
            "src.ui.app.requests.post",
            side_effect=requests.ConnectionError("Connection refused"),
        ):
            result = call_predict_api(sample_text, api_url)

        assert "error" in result
        assert "connection" in result["error"].lower() or "connect" in result["error"].lower()

    def test_server_error_returns_error_dict(self, sample_text, api_url):
        """call_predict_api returns error dict on HTTP 500."""
        from src.ui.app import call_predict_api

        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.json.return_value = {"error": "Internal server error"}

        with patch("src.ui.app.requests.post", return_value=mock_resp):
            result = call_predict_api(sample_text, api_url)

        assert "error" in result

    def test_empty_text_returns_error_without_calling_api(self, api_url):
        """call_predict_api returns error for empty text without making HTTP call."""
        from src.ui.app import call_predict_api

        with patch("src.ui.app.requests.post") as mock_post:
            result = call_predict_api("", api_url)

        mock_post.assert_not_called()
        assert "error" in result

    def test_whitespace_only_text_returns_error(self, api_url):
        """call_predict_api returns error for whitespace-only text."""
        from src.ui.app import call_predict_api

        with patch("src.ui.app.requests.post") as mock_post:
            result = call_predict_api("   \n\t  ", api_url)

        mock_post.assert_not_called()
        assert "error" in result

    def test_timeout_returns_error_dict(self, sample_text, api_url):
        """call_predict_api returns error dict on request timeout."""
        from src.ui.app import call_predict_api

        with patch(
            "src.ui.app.requests.post",
            side_effect=requests.Timeout("Request timed out"),
        ):
            result = call_predict_api(sample_text, api_url)

        assert "error" in result

    def test_result_contains_expected_keys(
        self, sample_text, api_url, mock_api_success_response
    ):
        """Successful result contains credibility_score, probability, and label."""
        from src.ui.app import call_predict_api

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = mock_api_success_response

        with patch("src.ui.app.requests.post", return_value=mock_resp):
            result = call_predict_api(sample_text, api_url)

        assert "credibility_score" in result
        assert "probability" in result
        assert "label" in result
