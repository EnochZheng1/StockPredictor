import pytest
import pandas as pd
from unittest.mock import patch
from services.data_service import prepare_data, DataBundle


class TestPrepareData:
    def test_returns_data_bundle(self, mock_yfinance):
        result = prepare_data("AAPL")
        assert isinstance(result, DataBundle)

    def test_correct_split_ratio(self, mock_yfinance):
        result = prepare_data("AAPL", test_ratio=0.2)
        total = len(result.X_train) + len(result.X_test)
        ratio = len(result.X_test) / total
        assert 0.15 < ratio < 0.25

    def test_no_nan_in_features(self, mock_yfinance):
        result = prepare_data("AAPL")
        assert not result.X_train.isnull().any().any()
        assert not result.X_test.isnull().any().any()

    def test_excludes_non_feature_cols(self, mock_yfinance):
        result = prepare_data("AAPL")
        assert "Close" not in result.feature_columns
        assert "Date" not in result.feature_columns

    def test_empty_df_raises(self):
        with patch(
            "services.data_service.get_historical_data",
            return_value=pd.DataFrame(),
        ):
            with pytest.raises(ValueError, match="No data"):
                prepare_data("INVALID")

    def test_none_df_raises(self):
        with patch(
            "services.data_service.get_historical_data", return_value=None
        ):
            with pytest.raises(ValueError, match="No data"):
                prepare_data("INVALID")

    def test_too_few_rows_raises(self):
        small = pd.DataFrame(
            {
                "Open": [1] * 10,
                "High": [2] * 10,
                "Low": [0.5] * 10,
                "Close": [1.5] * 10,
                "Adj Close": [1.5] * 10,
                "Volume": [100] * 10,
            }
        )
        small["Date"] = pd.bdate_range("2024-01-01", periods=10)
        with patch(
            "services.data_service.get_historical_data", return_value=small
        ):
            with pytest.raises(ValueError, match="Not enough data"):
                prepare_data("TINY")

    def test_feature_columns_are_numeric(self, mock_yfinance):
        import numpy as _np
        result = prepare_data("AAPL")
        for col in result.feature_columns:
            assert _np.issubdtype(result.X_train[col].dtype, _np.number)

    def test_has_feature_columns(self, mock_yfinance):
        result = prepare_data("AAPL")
        assert len(result.feature_columns) > 0

    def test_train_test_dates_match_split(self, mock_yfinance):
        result = prepare_data("AAPL")
        assert len(result.train_dates) == len(result.X_train)
        assert len(result.test_dates) == len(result.X_test)
