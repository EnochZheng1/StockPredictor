import logging
from fastapi import APIRouter, HTTPException
from typing import List, Dict
from pydantic import BaseModel

from services.data_service import prepare_data
from utils.models import get_model

logger = logging.getLogger(__name__)

router = APIRouter()


class PortfolioRequest(BaseModel):
    tickers: List[str]
    model_name: str
    steps: int = 30
    period: str = "5y"


class TickerResult(BaseModel):
    ticker: str
    current_price: float
    predicted_price: float
    predicted_change: float  # percentage
    rmse: float
    r2: float


class PortfolioResponse(BaseModel):
    results: List[TickerResult]
    model_name: str


@router.post("/portfolio", response_model=PortfolioResponse)
def portfolio_compare(request: PortfolioRequest):
    try:
        results = []
        model_display_name = request.model_name

        for ticker in request.tickers:
            try:
                data = prepare_data(ticker, period=request.period)
                model = get_model(request.model_name)
                model.train(data.X_train, data.y_train)
                metrics = model.evaluate(data.X_test, data.y_test)
                future = model.predict_future(data.X_test, steps=request.steps)
                model_display_name = model.get_name()

                current = float(data.y_test.iloc[-1])
                predicted = float(future[-1]) if future and len(future) > 0 else current
                change = ((predicted - current) / current) * 100 if current != 0 else 0.0

                results.append(TickerResult(
                    ticker=ticker.upper(),
                    current_price=round(current, 2),
                    predicted_price=round(predicted, 2),
                    predicted_change=round(change, 2),
                    rmse=round(metrics["rmse"], 4),
                    r2=round(metrics["r2"], 4),
                ))
            except Exception as e:
                logger.error("Portfolio %s failed: %s", ticker, e)

        results.sort(key=lambda r: r.predicted_change, reverse=True)
        return PortfolioResponse(results=results, model_name=model_display_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
