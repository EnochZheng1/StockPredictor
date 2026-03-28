import sys
import os
import pytest
import pandas as pd
import numpy as np

# Add backend to path so imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture
def sample_feature_data():
    """Generate synthetic stock-like data for testing models."""
    np.random.seed(42)
    n = 200
    dates = pd.bdate_range("2020-01-01", periods=n)
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

    # Time-ordered split
    split = int(len(df) * 0.8)
    return {
        "X_train": df.iloc[:split],
        "X_test": df.iloc[split:],
        "y_train": y.iloc[:split],
        "y_test": y.iloc[split:],
    }
