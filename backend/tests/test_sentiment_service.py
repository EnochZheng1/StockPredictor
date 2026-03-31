import pytest
from services.sentiment_service import get_news_sentiment, _clean_text, VADER_AVAILABLE
from unittest.mock import patch

pytestmark = pytest.mark.skipif(not VADER_AVAILABLE, reason="vaderSentiment not installed")


class TestCleanText:
    def test_removes_html(self):
        assert _clean_text("<b>Hello</b> world") == "Hello world"

    def test_normalizes_whitespace(self):
        assert _clean_text("  too   many  spaces  ") == "too many spaces"


class TestGetNewsSentiment:
    def test_returns_correct_structure(self, mock_yfinance):
        result = get_news_sentiment("AAPL")
        assert "ticker" in result
        assert "articles" in result
        assert "avg_sentiment" in result
        assert "total_articles" in result

    def test_labels_assigned(self, mock_yfinance):
        result = get_news_sentiment("AAPL")
        for article in result["articles"]:
            assert article["sentiment_label"] in {
                "positive",
                "negative",
                "neutral",
            }

    def test_respects_max_items(self, mock_yfinance):
        result = get_news_sentiment("AAPL", max_items=1)
        assert len(result["articles"]) <= 1

    def test_error_handled(self):
        with patch(
            "services.sentiment_service.yf.Ticker",
            side_effect=Exception("fail"),
        ):
            result = get_news_sentiment("AAPL")
            assert "error" in result

    def test_vader_unavailable(self, monkeypatch):
        import services.sentiment_service as mod

        monkeypatch.setattr(mod, "VADER_AVAILABLE", False)
        result = get_news_sentiment("AAPL")
        assert "error" in result
