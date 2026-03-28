import numpy as np
import pytest

from utils.models.linear_regression import LinearRegressionModel
from utils.models.random_forest import RandomForestModel
from utils.models.xgboost_model import XGBoostModel
from utils.models.polynomial_regression import PolynomialRegressionModel
from utils.models.arima_model import ARIMAModel


class TestFeatureBasedModels:
    """Tests for models that use feature matrices (not time-series only)."""

    @pytest.mark.parametrize("ModelClass", [
        LinearRegressionModel,
        RandomForestModel,
        XGBoostModel,
        PolynomialRegressionModel,
    ])
    def test_train_predict_evaluate(self, sample_feature_data, ModelClass):
        model = ModelClass()
        d = sample_feature_data

        model.train(d["X_train"], d["y_train"])
        preds = model.predict(d["X_test"])

        assert len(preds) == len(d["X_test"])
        assert not np.any(np.isnan(preds))

    @pytest.mark.parametrize("ModelClass", [
        LinearRegressionModel,
        RandomForestModel,
        XGBoostModel,
        PolynomialRegressionModel,
    ])
    def test_evaluate_returns_metrics(self, sample_feature_data, ModelClass):
        model = ModelClass()
        d = sample_feature_data

        model.train(d["X_train"], d["y_train"])
        metrics = model.evaluate(d["X_test"], d["y_test"])

        assert "rmse" in metrics
        assert "mae" in metrics
        assert "r2" in metrics
        assert metrics["rmse"] >= 0
        assert metrics["mae"] >= 0

    @pytest.mark.parametrize("ModelClass", [
        LinearRegressionModel,
        RandomForestModel,
        XGBoostModel,
        PolynomialRegressionModel,
    ])
    def test_predict_future(self, sample_feature_data, ModelClass):
        model = ModelClass()
        d = sample_feature_data

        model.train(d["X_train"], d["y_train"])
        future = model.predict_future(d["X_test"], steps=10)

        assert len(future) == 10
        assert all(isinstance(v, float) for v in future)

    @pytest.mark.parametrize("ModelClass", [
        LinearRegressionModel,
        RandomForestModel,
        XGBoostModel,
        PolynomialRegressionModel,
    ])
    def test_get_name(self, ModelClass):
        model = ModelClass()
        name = model.get_name()
        assert isinstance(name, str)
        assert len(name) > 0


class TestARIMAModel:
    def test_train_predict(self, sample_feature_data):
        model = ARIMAModel(order=(2, 1, 0))
        d = sample_feature_data

        model.train(d["X_train"], d["y_train"])
        preds = model.predict(d["X_test"])

        assert len(preds) == len(d["X_test"])

    def test_predict_future(self, sample_feature_data):
        model = ARIMAModel(order=(2, 1, 0))
        d = sample_feature_data

        model.train(d["X_train"], d["y_train"])
        future = model.predict_future(d["X_test"], steps=5)

        assert len(future) == 5

    def test_get_name(self):
        assert ARIMAModel().get_name() == "ARIMA"
