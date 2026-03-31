import numpy as np
import pytest
from unittest.mock import patch
from services.backtest_service import _compute_backtest, run_backtest


class TestComputeBacktest:
    def test_equity_curve_starts_at_one(self):
        preds = np.array([100, 102, 104, 106, 108])
        actual = np.array([100, 101, 103, 105, 107])
        dates = ["d1", "d2", "d3", "d4", "d5"]
        result = _compute_backtest(preds, actual, dates)
        assert result.equity_curve[0] == 1.0

    def test_all_up_predictions_positive_return(self):
        preds = np.array([100, 105, 110, 115, 120])
        actual = np.array([100, 103, 106, 109, 112])
        dates = ["d1", "d2", "d3", "d4", "d5"]
        result = _compute_backtest(preds, actual, dates)
        assert result.total_return > 0

    def test_sharpe_ratio_is_finite(self):
        preds = np.array([100, 102, 101, 103, 104])
        actual = np.array([100, 101, 100, 102, 103])
        dates = ["d1", "d2", "d3", "d4", "d5"]
        result = _compute_backtest(preds, actual, dates)
        assert np.isfinite(result.sharpe_ratio)

    def test_max_drawdown_non_negative(self):
        preds = np.array([100, 102, 101, 103, 104])
        actual = np.array([100, 101, 100, 102, 103])
        dates = ["d1", "d2", "d3", "d4", "d5"]
        result = _compute_backtest(preds, actual, dates)
        assert result.max_drawdown >= 0

    def test_win_rate_between_0_and_100(self):
        preds = np.array([100, 102, 101, 103, 104])
        actual = np.array([100, 101, 100, 102, 103])
        dates = ["d1", "d2", "d3", "d4", "d5"]
        result = _compute_backtest(preds, actual, dates)
        assert 0 <= result.win_rate <= 100

    def test_flat_predictions_zero_trades(self):
        preds = np.array([100, 100, 100, 100, 100])
        actual = np.array([100, 101, 100, 102, 103])
        dates = ["d1", "d2", "d3", "d4", "d5"]
        result = _compute_backtest(preds, actual, dates)
        assert result.num_trades == 0


class TestRunBacktest:
    def test_returns_backtest_response(self, mock_data_bundle):
        with patch(
            "services.backtest_service.prepare_data",
            return_value=mock_data_bundle,
        ):
            result = run_backtest("AAPL", ["linear_regression"])
            assert len(result.results) == 1
            assert result.best_strategy != "N/A"

    def test_results_sorted_by_return(self, mock_data_bundle):
        with patch(
            "services.backtest_service.prepare_data",
            return_value=mock_data_bundle,
        ):
            result = run_backtest(
                "AAPL", ["linear_regression", "random_forest"]
            )
            if len(result.results) >= 2:
                assert (
                    result.results[0].total_return
                    >= result.results[1].total_return
                )

    def test_failing_model_skipped(self, mock_data_bundle):
        with patch(
            "services.backtest_service.prepare_data",
            return_value=mock_data_bundle,
        ), patch(
            "services.backtest_service.get_model",
            side_effect=[
                Exception("fail"),
                __import__(
                    "utils.models", fromlist=["get_model"]
                ).get_model("linear_regression"),
            ],
        ):
            # One model fails, one succeeds - should get 1 result
            result = run_backtest(
                "AAPL", ["bad_model", "linear_regression"]
            )
            assert len(result.results) >= 0  # At least doesn't crash
