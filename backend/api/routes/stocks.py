from fastapi import APIRouter, HTTPException
import numpy as np
import pandas as pd

from utils.data_fetcher import get_historical_data
from utils.data_analysis import technical_indictors_calculation
from utils.models import list_models, get_model_params
from services.ensemble_service import list_ensemble_methods
from services.db_service import get_history
from api.schemas import StockDataResponse

router = APIRouter()


@router.get("/stocks/{ticker}", response_model=StockDataResponse)
def get_stock_data(ticker: str, period: str = "5y"):
    try:
        df = get_historical_data(ticker, period=period)
        df = technical_indictors_calculation(df)

        if "Date" not in df.columns and df.index.name == "Date":
            df = df.reset_index()

        # Flatten multi-level columns if needed
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] if col[1] == "" else f"{col[0]}_{col[1]}" for col in df.columns]

        dates = [str(d.date()) if hasattr(d, "date") else str(d) for d in df["Date"]]

        def safe_list(series):
            return [None if (isinstance(v, float) and np.isnan(v)) else float(v) for v in series]

        # Collect numeric indicator columns
        skip = {"Open", "High", "Low", "Close", "Adj Close", "Volume", "Date"}
        indicators = {}
        for col in df.columns:
            if col not in skip and df[col].dtype in [np.float64, np.int64, np.float32]:
                indicators[col] = safe_list(df[col])

        return StockDataResponse(
            ticker=ticker.upper(),
            dates=dates,
            open_prices=safe_list(df["Open"]),
            high_prices=safe_list(df["High"]),
            low_prices=safe_list(df["Low"]),
            close_prices=safe_list(df["Close"]),
            volume=safe_list(df["Volume"]),
            indicators=indicators,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/models")
def get_available_models():
    return {
        "models": list_models(),
        "ensemble_methods": list_ensemble_methods(),
        "model_params": get_model_params(),
    }


@router.get("/history")
def get_prediction_history(ticker: str = None, limit: int = 50):
    return {"history": get_history(ticker=ticker, limit=limit)}
