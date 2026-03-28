from pydantic import BaseModel
from typing import List, Dict, Optional


class PredictionRequest(BaseModel):
    ticker: str
    model_name: str
    steps: int = 30
    period: str = "5y"


class ComparisonRequest(BaseModel):
    ticker: str
    model_names: List[str]
    steps: int = 30
    period: str = "5y"
    ensemble_methods: List[str] = []


class MetricsResponse(BaseModel):
    rmse: float
    mae: float
    r2: float


class PredictionResponse(BaseModel):
    model_name: str
    metrics: MetricsResponse
    test_predictions: List[float]
    test_dates: List[str]
    future_predictions: List[float]
    future_dates: List[str]
    feature_importance: Optional[Dict[str, float]] = None


class ComparisonResponse(BaseModel):
    results: List[PredictionResponse]
    summary: List[Dict[str, float]]
    best_model: str
    errors: List[str] = []


class StockDataResponse(BaseModel):
    ticker: str
    dates: List[str]
    close_prices: List[float]
    indicators: Dict[str, List[Optional[float]]]
