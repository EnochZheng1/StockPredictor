import pytest
from unittest.mock import patch
from services.prediction_service import run_prediction, PredictionResult


class TestRunPrediction:
    def test_returns_prediction_result(self, mock_data_bundle):
        with patch(
            "services.prediction_service.prepare_data",
            return_value=mock_data_bundle,
        ):
            result = run_prediction("AAPL", "linear_regression", steps=5)
            assert isinstance(result, PredictionResult)

    def test_correct_test_predictions_length(self, mock_data_bundle):
        with patch(
            "services.prediction_service.prepare_data",
            return_value=mock_data_bundle,
        ):
            result = run_prediction("AAPL", "linear_regression", steps=5)
            assert len(result.test_predictions) == len(mock_data_bundle.X_test)

    def test_correct_future_length(self, mock_data_bundle):
        with patch(
            "services.prediction_service.prepare_data",
            return_value=mock_data_bundle,
        ):
            result = run_prediction("AAPL", "linear_regression", steps=10)
            assert len(result.future_predictions) == 10
            assert len(result.future_dates) == 10

    def test_invalid_model_raises(self, mock_data_bundle):
        with patch(
            "services.prediction_service.prepare_data",
            return_value=mock_data_bundle,
        ):
            with pytest.raises(ValueError, match="Unknown model"):
                run_prediction("AAPL", "nonexistent_model")

    def test_predictions_are_floats(self, mock_data_bundle):
        with patch(
            "services.prediction_service.prepare_data",
            return_value=mock_data_bundle,
        ):
            result = run_prediction("AAPL", "random_forest", steps=3)
            assert all(isinstance(v, float) for v in result.test_predictions)
            assert all(
                isinstance(v, float) for v in result.future_predictions
            )

    def test_future_dates_are_strings(self, mock_data_bundle):
        with patch(
            "services.prediction_service.prepare_data",
            return_value=mock_data_bundle,
        ):
            result = run_prediction("AAPL", "linear_regression", steps=5)
            assert all(isinstance(d, str) for d in result.future_dates)
