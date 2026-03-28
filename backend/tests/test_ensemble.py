import numpy as np
import pytest

from services.prediction_service import PredictionResult
from services.ensemble_service import (
    simple_average_ensemble,
    weighted_average_ensemble,
    run_ensembles,
    list_ensemble_methods,
)
from services.data_service import DataBundle
import pandas as pd


@pytest.fixture
def mock_base_results():
    """Create mock PredictionResult objects for ensemble testing."""
    dates = ["2024-01-01", "2024-01-02", "2024-01-03"]
    future_dates = ["2024-01-04", "2024-01-05"]

    r1 = PredictionResult(
        model_name="Model A",
        metrics={"rmse": 2.0, "mae": 1.5, "r2": 0.8},
        test_predictions=[100.0, 102.0, 104.0],
        test_dates=dates,
        future_predictions=[106.0, 108.0],
        future_dates=future_dates,
        model_key="linear_regression",
    )
    r2 = PredictionResult(
        model_name="Model B",
        metrics={"rmse": 4.0, "mae": 3.0, "r2": 0.6},
        test_predictions=[101.0, 103.0, 105.0],
        test_dates=dates,
        future_predictions=[107.0, 109.0],
        future_dates=future_dates,
        model_key="random_forest",
    )
    return [r1, r2]


@pytest.fixture
def mock_data_bundle():
    """Minimal DataBundle for ensemble evaluation."""
    y_test = pd.Series([100.5, 102.5, 104.5])
    return DataBundle(
        df_full=pd.DataFrame(),
        X_train=pd.DataFrame(np.random.randn(50, 3)),
        X_test=pd.DataFrame(np.random.randn(3, 3)),
        y_train=pd.Series(np.random.randn(50) + 100),
        y_test=y_test,
        feature_columns=["a", "b", "c"],
        train_dates=pd.Series(pd.bdate_range("2023-01-01", periods=50)),
        test_dates=pd.Series(pd.bdate_range("2024-01-01", periods=3)),
    )


class TestSimpleAverage:
    def test_averages_predictions(self, mock_base_results, mock_data_bundle):
        result = simple_average_ensemble(
            mock_base_results, mock_data_bundle, 2,
            ["2024-01-01", "2024-01-02", "2024-01-03"],
            ["2024-01-04", "2024-01-05"],
        )

        # Average of [100, 101], [102, 103], [104, 105]
        assert result.test_predictions == pytest.approx([100.5, 102.5, 104.5])
        assert result.future_predictions == pytest.approx([106.5, 108.5])
        assert result.model_name == "Ensemble (Average)"

    def test_returns_valid_metrics(self, mock_base_results, mock_data_bundle):
        result = simple_average_ensemble(
            mock_base_results, mock_data_bundle, 2,
            ["2024-01-01", "2024-01-02", "2024-01-03"],
            ["2024-01-04", "2024-01-05"],
        )
        assert "rmse" in result.metrics
        assert "mae" in result.metrics
        assert "r2" in result.metrics


class TestWeightedAverage:
    def test_weights_by_inverse_rmse(self, mock_base_results, mock_data_bundle):
        result = weighted_average_ensemble(
            mock_base_results, mock_data_bundle, 2,
            ["2024-01-01", "2024-01-02", "2024-01-03"],
            ["2024-01-04", "2024-01-05"],
        )

        # Model A has RMSE=2, Model B has RMSE=4
        # Weights: 1/2=0.5, 1/4=0.25 -> normalized: 2/3, 1/3
        # So weighted avg should be closer to Model A
        assert result.test_predictions[0] < 100.5  # closer to Model A's 100
        assert result.model_name == "Ensemble (Weighted)"


class TestRunEnsembles:
    def test_skips_with_one_model(self, mock_data_bundle):
        single = [PredictionResult(
            model_name="Solo", metrics={"rmse": 1}, test_predictions=[1],
            test_dates=["d"], future_predictions=[2], future_dates=["d"],
        )]
        results = run_ensembles(["ensemble_average"], single, mock_data_bundle, 1, ["d"], ["d"])
        assert results == []

    def test_returns_results_for_valid_methods(self, mock_base_results, mock_data_bundle):
        results = run_ensembles(
            ["ensemble_average", "ensemble_weighted"],
            mock_base_results, mock_data_bundle, 2,
            ["2024-01-01", "2024-01-02", "2024-01-03"],
            ["2024-01-04", "2024-01-05"],
        )
        assert len(results) == 2

    def test_skips_unknown_methods(self, mock_base_results, mock_data_bundle):
        results = run_ensembles(
            ["nonexistent_method"],
            mock_base_results, mock_data_bundle, 2,
            ["d1", "d2", "d3"], ["d4", "d5"],
        )
        assert results == []


class TestListEnsembleMethods:
    def test_returns_all_methods(self):
        methods = list_ensemble_methods()
        assert "ensemble_average" in methods
        assert "ensemble_weighted" in methods
        assert "ensemble_stacking" in methods
