from fastapi import APIRouter, HTTPException
import numpy as np

from utils.data_fetcher import get_historical_data
from utils.data_analysis import technical_indictors_calculation
from utils.models import list_models
from api.schemas import StockDataResponse

router = APIRouter()


@router.get("/stocks/{ticker}", response_model=StockDataResponse)
def get_stock_data(ticker: str):
    try:
        df = get_historical_data(ticker)
        df = technical_indictors_calculation(df)

        if "Date" not in df.columns and df.index.name == "Date":
            df = df.reset_index()

        dates = [str(d.date()) if hasattr(d, "date") else str(d) for d in df["Date"]]
        close_prices = df["Close"].tolist()

        # Collect numeric indicator columns
        skip = {"Open", "High", "Low", "Close", "Adj Close", "Volume", "Date"}
        indicators = {}
        for col in df.columns:
            if col not in skip and df[col].dtype in [np.float64, np.int64, np.float32]:
                indicators[col] = [
                    None if (isinstance(v, float) and np.isnan(v)) else float(v)
                    for v in df[col]
                ]

        return StockDataResponse(
            ticker=ticker.upper(),
            dates=dates,
            close_prices=close_prices,
            indicators=indicators,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/models")
def get_available_models():
    return {"models": list_models()}
