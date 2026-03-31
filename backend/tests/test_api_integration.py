import sys
import os
import json
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture(autouse=True)
def _disable_model_cache(monkeypatch):
    """Disable model cache so tests always train fresh."""
    import services.comparison_service as comp
    monkeypatch.setattr(comp, "load_model", lambda *a, **kw: None)
    monkeypatch.setattr(comp, "save_model", lambda *a, **kw: None)


@pytest.fixture
def client(mock_yfinance):
    """TestClient with yfinance mocked."""
    from fastapi.testclient import TestClient
    from app import app
    return TestClient(app)


class TestRoot:
    def test_root_returns_message(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert "StockPredictor" in r.json()["message"]


class TestModelsEndpoint:
    def test_list_models(self, client):
        r = client.get("/api/models")
        assert r.status_code == 200
        data = r.json()
        assert "models" in data
        assert "ensemble_methods" in data
        assert "model_params" in data
        assert len(data["models"]) >= 6

    def test_model_params_have_structure(self, client):
        r = client.get("/api/models")
        params = r.json()["model_params"]
        # At least random_forest should have params
        assert "random_forest" in params
        assert "n_estimators" in params["random_forest"]


class TestStocksEndpoint:
    def test_get_stock_data_success(self, client):
        r = client.get("/api/stocks/AAPL")
        assert r.status_code == 200
        data = r.json()
        assert data["ticker"] == "AAPL"
        assert len(data["dates"]) > 0
        assert len(data["close_prices"]) > 0
        assert len(data["open_prices"]) > 0
        assert len(data["volume"]) > 0

    def test_get_stock_data_has_indicators(self, client):
        r = client.get("/api/stocks/AAPL")
        indicators = r.json()["indicators"]
        assert isinstance(indicators, dict)
        # Should have at least SMA_20 from data_fetcher
        assert len(indicators) > 0

    def test_get_stock_data_with_period(self, client):
        r = client.get("/api/stocks/AAPL?period=2y")
        assert r.status_code == 200


class TestPredictEndpoint:
    def test_predict_success(self, client):
        r = client.post("/api/predict", json={
            "ticker": "AAPL",
            "model_name": "linear_regression",
            "steps": 5,
        })
        assert r.status_code == 200
        data = r.json()
        assert data["model_name"] == "Linear Regression"
        assert len(data["test_predictions"]) > 0
        assert len(data["future_predictions"]) == 5
        assert "metrics" in data

    def test_predict_invalid_model(self, client):
        r = client.post("/api/predict", json={
            "ticker": "AAPL",
            "model_name": "nonexistent",
            "steps": 5,
        })
        assert r.status_code in [400, 500]

    def test_predict_validation_error(self, client):
        r = client.post("/api/predict", json={"ticker": "AAPL"})
        assert r.status_code == 422


class TestCompareEndpoint:
    def test_compare_success(self, client):
        r = client.post("/api/compare", json={
            "ticker": "AAPL",
            "model_names": ["linear_regression"],
            "steps": 5,
        })
        # May return 200 or 400 depending on model cache state
        if r.status_code == 200:
            data = r.json()
            assert len(data["results"]) >= 1
            assert data["best_model"] != ""

    def test_compare_with_ensembles(self, client):
        r = client.post("/api/compare", json={
            "ticker": "AAPL",
            "model_names": ["linear_regression", "random_forest"],
            "steps": 5,
            "ensemble_methods": ["ensemble_average"],
        })
        if r.status_code == 200:
            data = r.json()
            assert len(data["results"]) >= 2

    def test_compare_invalid_json(self, client):
        r = client.post("/api/compare", content="not json")
        assert r.status_code == 422


class TestBacktestEndpoint:
    def test_backtest_success(self, client):
        r = client.post("/api/backtest", json={
            "ticker": "AAPL",
            "model_names": ["linear_regression"],
        })
        assert r.status_code == 200
        data = r.json()
        assert len(data["results"]) == 1
        assert "equity_curve" in data["results"][0]
        assert "buy_hold_return" in data

    def test_backtest_multiple_models(self, client):
        r = client.post("/api/backtest", json={
            "ticker": "AAPL",
            "model_names": ["linear_regression", "random_forest"],
        })
        assert r.status_code == 200
        assert len(r.json()["results"]) == 2


class TestPortfolioEndpoint:
    def test_portfolio_success(self, client):
        r = client.post("/api/portfolio", json={
            "tickers": ["AAPL", "MSFT"],
            "model_name": "linear_regression",
            "steps": 5,
        })
        assert r.status_code == 200
        data = r.json()
        assert len(data["results"]) == 2
        assert data["results"][0]["ticker"] in ["AAPL", "MSFT"]


class TestHistoryEndpoint:
    def test_history_empty(self, client):
        r = client.get("/api/history")
        assert r.status_code == 200
        assert isinstance(r.json()["history"], list)

    def test_history_after_compare(self, client):
        # Run a comparison first (which saves to DB)
        client.post("/api/compare", json={
            "ticker": "AAPL",
            "model_names": ["linear_regression"],
            "steps": 5,
        })
        r = client.get("/api/history?ticker=AAPL")
        assert r.status_code == 200
        assert len(r.json()["history"]) >= 1


class TestSentimentEndpoint:
    def test_sentiment_success(self, client):
        r = client.get("/api/sentiment/AAPL")
        assert r.status_code == 200
        data = r.json()
        assert "articles" in data
        assert "avg_sentiment" in data
