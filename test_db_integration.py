import unittest
from unittest.mock import Mock, patch

import requests

from src import db_integration


class TestDbIntegration(unittest.TestCase):
    @patch("src.db_integration.requests.post")
    def test_send_prediction_data_success(self, mock_post):
        db_integration.config.DOTNET_WEBHOOK_URL = "https://example.com/webhook"
        db_integration.config.INTERNAL_API_KEY = "test-key"
        mock_response = Mock(status_code=201, text="created")
        mock_post.return_value = mock_response

        result = db_integration.send_prediction_data({"id": 1})

        self.assertTrue(result)
        mock_post.assert_called_once()

    @patch("src.db_integration.requests.post")
    def test_send_intermediate_data_skips_without_api_key(self, mock_post):
        db_integration.config.DOTNET_PREDICTION_URL = "https://example.com/prediction"
        db_integration.config.INTERNAL_API_KEY = None

        result = db_integration.send_intermediate_data({"value": "sample"})

        self.assertFalse(result)
        mock_post.assert_not_called()

    @patch(
        "src.db_integration.requests.post",
        side_effect=requests.RequestException("boom"),
    )
    def test_send_prediction_data_handles_exceptions(self, mock_post):
        db_integration.config.DOTNET_WEBHOOK_URL = "https://example.com/webhook"
        db_integration.config.INTERNAL_API_KEY = "test-key"

        result = db_integration.send_prediction_data({"id": 2})

        self.assertFalse(result)
        mock_post.assert_called_once()


if __name__ == "__main__":
    unittest.main()
