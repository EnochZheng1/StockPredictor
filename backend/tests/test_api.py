import sys
import os
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app import app

client = TestClient(app)


class TestRootEndpoint:
    def test_root_returns_message(self):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "StockPredictor" in data["message"]


class TestModelsEndpoint:
    def test_list_models(self):
        response = client.get("/api/models")
        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        assert "ensemble_methods" in data
        assert isinstance(data["models"], list)
        assert len(data["models"]) >= 6  # at least 6 models (Prophet may be missing)
        assert "linear_regression" in data["models"]
        assert "random_forest" in data["models"]

    def test_ensemble_methods_listed(self):
        response = client.get("/api/models")
        data = response.json()
        assert "ensemble_average" in data["ensemble_methods"]
        assert "ensemble_weighted" in data["ensemble_methods"]
        assert "ensemble_stacking" in data["ensemble_methods"]


class TestCompareEndpoint:
    def test_empty_model_list_returns_error(self):
        response = client.post("/api/compare", json={
            "ticker": "AAPL",
            "model_names": [],
            "steps": 5,
        })
        assert response.status_code in [400, 500]

    def test_rejects_invalid_json(self):
        response = client.post("/api/compare", content="not json")
        assert response.status_code == 422


class TestPredictEndpoint:
    def test_rejects_unknown_model(self):
        response = client.post("/api/predict", json={
            "ticker": "AAPL",
            "model_name": "nonexistent_model",
            "steps": 5,
        })
        assert response.status_code in [400, 500]


class TestStocksEndpoint:
    def test_rejects_invalid_ticker(self):
        response = client.get("/api/stocks/ZZZZZZZZZ999")
        # yfinance may return empty data or error
        assert response.status_code in [200, 400]
