import sys
import os
import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from dataclasses import dataclass

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# Shared data fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_feature_data():
    """Synthetic stock-like data for model unit tests."""
    np.random.seed(42)
    n = 200
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)

    df = pd.DataFrame({
        "Open": close + np.random.randn(n) * 0.3,
        "High": close + abs(np.random.randn(n) * 0.5),
        "Low": close - abs(np.random.randn(n) * 0.5),
        "SMA_20": pd.Series(close).rolling(20).mean(),
        "EMA_20": pd.Series(close).ewm(span=20).mean(),
        "RSI": 50 + np.random.randn(n) * 10,
        "MACD": np.random.randn(n) * 0.5,
        "Momentum": np.random.randn(n) * 2,
    })
    df = df.dropna().reset_index(drop=True)
    y = pd.Series(close[:len(df)], name="Close")

    split = int(len(df) * 0.8)
    return {
        "X_train": df.iloc[:split],
        "X_test": df.iloc[split:],
        "y_train": y.iloc[:split],
        "y_test": y.iloc[split:],
    }


@pytest.fixture
def mock_stock_dataframe():
    """300-row DataFrame simulating yfinance output with all required columns."""
    np.random.seed(42)
    n = 300
    dates = pd.bdate_range("2019-01-02", periods=n)
    close = 100 + np.cumsum(np.random.randn(n) * 1.0)
    high = close + abs(np.random.randn(n) * 1.5)
    low = close - abs(np.random.randn(n) * 1.5)
    open_ = close + np.random.randn(n) * 0.5
    volume = np.random.randint(1_000_000, 50_000_000, n).astype(float)

    df = pd.DataFrame({
        "Open": open_,
        "High": high,
        "Low": low,
        "Close": close,
        "Adj Close": close * 0.99,
        "Volume": volume,
    }, index=dates)
    df.index.name = "Date"
    df["Date"] = df.index
    df["SMA_20"] = df["Adj Close"].rolling(window=20).mean()
    return df


@pytest.fixture
def mock_data_bundle(sample_feature_data):
    """Pre-built DataBundle for service tests."""
    from services.data_service import DataBundle
    d = sample_feature_data
    n_train = len(d["X_train"])
    n_test = len(d["X_test"])
    return DataBundle(
        df_full=pd.DataFrame(),
        X_train=d["X_train"],
        X_test=d["X_test"],
        y_train=d["y_train"],
        y_test=d["y_test"],
        feature_columns=list(d["X_train"].columns),
        train_dates=pd.Series(pd.bdate_range("2020-01-01", periods=n_train)),
        test_dates=pd.Series(pd.bdate_range("2024-01-01", periods=n_test)),
    )


# ---------------------------------------------------------------------------
# Database isolation (autouse)
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _isolate_db(tmp_path, monkeypatch):
    """Every test uses a temp database, never the production one."""
    db_path = str(tmp_path / "test.db")
    import services.db_service as db_mod
    monkeypatch.setattr(db_mod, "DB_PATH", db_path)
    # Re-initialize with temp path
    db_mod.init_db()


# ---------------------------------------------------------------------------
# Model storage isolation
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_model_dir(tmp_path, monkeypatch):
    model_dir = str(tmp_path / "models")
    os.makedirs(model_dir, exist_ok=True)
    import services.model_storage as storage_mod
    monkeypatch.setattr(storage_mod, "SAVE_DIR", model_dir)
    return model_dir


# ---------------------------------------------------------------------------
# yfinance mock
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_yfinance(mock_stock_dataframe):
    """Patches yfinance.download and yfinance.Ticker globally."""
    mock_ticker = MagicMock()
    mock_ticker.news = [
        {
            "title": "Stock rises on strong earnings",
            "publisher": "Reuters",
            "link": "https://example.com/1",
            "providerPublishTime": 1700000000,
        },
        {
            "title": "Market faces downturn pressure",
            "publisher": "Bloomberg",
            "link": "https://example.com/2",
            "providerPublishTime": 1700001000,
        },
    ]
    mock_ticker.fast_info = MagicMock()
    mock_ticker.fast_info.last_price = 150.25
    mock_ticker.fast_info.previous_close = 148.50

    with patch("utils.data_fetcher.yf.download", return_value=mock_stock_dataframe) as mock_dl, \
         patch("services.sentiment_service.yf.Ticker", return_value=mock_ticker) as mock_tk:
        # Also clear the data cache so mocks take effect
        import utils.data_fetcher as fetcher
        fetcher._cache.clear()
        yield {"download": mock_dl, "Ticker": mock_tk, "ticker_instance": mock_ticker}
