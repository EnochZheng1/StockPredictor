import logging
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List

from utils.data_fetcher import get_historical_data
from utils.data_analysis import technical_indictors_calculation

logger = logging.getLogger(__name__)


@dataclass
class DataBundle:
    df_full: pd.DataFrame
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    feature_columns: List[str]
    train_dates: pd.Series
    test_dates: pd.Series


# Columns to exclude from features
NON_FEATURE_COLS = {"Date", "Adj Close", "Close", "Volume_diff", "Chikou_span"}


def prepare_data(ticker: str, period: str = "5y", test_ratio: float = 0.2) -> DataBundle:
    """Fetch stock data, compute indicators, and split into train/test."""
    df = get_historical_data(ticker, period=period)

    if df is None or df.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'. Check that it is a valid symbol.")

    df = technical_indictors_calculation(df)

    # Flatten multi-level columns if yfinance returns them
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if col[1] == "" else f"{col[0]}_{col[1]}" for col in df.columns]

    # Ensure Date column exists
    if "Date" not in df.columns and df.index.name == "Date":
        df = df.reset_index()

    if "Close" not in df.columns:
        raise ValueError(f"Data for '{ticker}' does not contain a 'Close' column.")

    # Drop rows with NaN from indicator warm-up periods
    df = df.dropna().reset_index(drop=True)

    if len(df) < 50:
        raise ValueError(f"Not enough data for '{ticker}' after indicator warm-up ({len(df)} rows). Need at least 50.")

    # Determine feature columns (numeric only, exclude target and non-features)
    feature_columns = [
        col for col in df.select_dtypes(include=[np.number]).columns
        if col not in NON_FEATURE_COLS
    ]

    if not feature_columns:
        raise ValueError(f"No numeric feature columns found for '{ticker}'.")

    X = df[feature_columns]
    y = df["Close"]
    dates = df["Date"] if "Date" in df.columns else pd.Series(df.index)

    # Time-ordered split (no shuffle!)
    split_idx = int(len(df) * (1 - test_ratio))

    logger.info("Data for %s: %d total rows, %d features, split at %d", ticker, len(df), len(feature_columns), split_idx)

    return DataBundle(
        df_full=df,
        X_train=X.iloc[:split_idx],
        X_test=X.iloc[split_idx:],
        y_train=y.iloc[:split_idx],
        y_test=y.iloc[split_idx:],
        feature_columns=feature_columns,
        train_dates=dates.iloc[:split_idx],
        test_dates=dates.iloc[split_idx:],
    )
