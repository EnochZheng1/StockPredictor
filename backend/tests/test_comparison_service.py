import pytest
from unittest.mock import patch, MagicMock
from services.comparison_service import run_comparison, ComparisonResult


class TestRunComparison:
    def test_returns_comparison_result(self, mock_data_bundle):
        with patch(
            "services.comparison_service.prepare_data",
            return_value=mock_data_bundle,
        ), patch(
            "services.comparison_service.load_model", return_value=None
        ), patch(
            "services.comparison_service.save_model"
        ), patch(
            "services.comparison_service.save_prediction"
        ):
            result = run_comparison(
                "AAPL", ["linear_regression"], steps=5
            )
            assert isinstance(result, ComparisonResult)
            assert len(result.results) == 1

    def test_multiple_models(self, mock_data_bundle):
        with patch(
            "services.comparison_service.prepare_data",
            return_value=mock_data_bundle,
        ), patch(
            "services.comparison_service.load_model", return_value=None
        ), patch(
            "services.comparison_service.save_model"
        ), patch(
            "services.comparison_service.save_prediction"
        ):
            result = run_comparison(
                "AAPL",
                ["linear_regression", "random_forest"],
                steps=5,
            )
            assert len(result.results) == 2

    def test_summary_sorted_by_rmse(self, mock_data_bundle):
        with patch(
            "services.comparison_service.prepare_data",
            return_value=mock_data_bundle,
        ), patch(
            "services.comparison_service.load_model", return_value=None
        ), patch(
            "services.comparison_service.save_model"
        ), patch(
            "services.comparison_service.save_prediction"
        ):
            result = run_comparison(
                "AAPL",
                ["linear_regression", "random_forest"],
                steps=5,
            )
            if len(result.summary) >= 2:
                assert result.summary[0]["rmse"] <= result.summary[1]["rmse"]

    def test_best_model_is_lowest_rmse(self, mock_data_bundle):
        with patch(
            "services.comparison_service.prepare_data",
            return_value=mock_data_bundle,
        ), patch(
            "services.comparison_service.load_model", return_value=None
        ), patch(
            "services.comparison_service.save_model"
        ), patch(
            "services.comparison_service.save_prediction"
        ):
            result = run_comparison(
                "AAPL",
                ["linear_regression", "random_forest"],
                steps=5,
            )
            assert result.best_model == result.summary[0]["model_name"]

    def test_with_ensemble_methods(self, mock_data_bundle):
        with patch(
            "services.comparison_service.prepare_data",
            return_value=mock_data_bundle,
        ), patch(
            "services.comparison_service.load_model", return_value=None
        ), patch(
            "services.comparison_service.save_model"
        ), patch(
            "services.comparison_service.save_prediction"
        ):
            result = run_comparison(
                "AAPL",
                ["linear_regression", "random_forest"],
                steps=5,
                ensemble_methods=["ensemble_average"],
            )
            assert len(result.results) == 3  # 2 models + 1 ensemble

    def test_failing_model_in_errors(self, mock_data_bundle):
        with patch(
            "services.comparison_service.prepare_data",
            return_value=mock_data_bundle,
        ), patch(
            "services.comparison_service.load_model", return_value=None
        ), patch(
            "services.comparison_service.save_model"
        ), patch(
            "services.comparison_service.save_prediction"
        ):
            result = run_comparison(
                "AAPL",
                ["nonexistent_model", "linear_regression"],
                steps=5,
            )
            assert len(result.errors) >= 1
            assert len(result.results) >= 1

    def test_saves_prediction_to_db(self, mock_data_bundle):
        mock_save = MagicMock()
        with patch(
            "services.comparison_service.prepare_data",
            return_value=mock_data_bundle,
        ), patch(
            "services.comparison_service.load_model", return_value=None
        ), patch(
            "services.comparison_service.save_model"
        ), patch(
            "services.comparison_service.save_prediction", mock_save
        ):
            run_comparison("AAPL", ["linear_regression"], steps=5)
            mock_save.assert_called_once()

    def test_feature_importance_for_tree_models(self, mock_data_bundle):
        with patch(
            "services.comparison_service.prepare_data",
            return_value=mock_data_bundle,
        ), patch(
            "services.comparison_service.load_model", return_value=None
        ), patch(
            "services.comparison_service.save_model"
        ), patch(
            "services.comparison_service.save_prediction"
        ):
            result = run_comparison(
                "AAPL", ["random_forest"], steps=5
            )
            assert result.results[0].feature_importance is not None
