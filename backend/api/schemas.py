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
    model_params: Dict[str, Dict] = {}  # e.g. {"random_forest": {"n_estimators": 200}}


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


class BacktestRequest(BaseModel):
    ticker: str
    model_names: List[str]
    period: str = "5y"
    model_params: Dict[str, Dict] = {}


class BacktestResultResponse(BaseModel):
    model_name: str
    total_return: float
    buy_hold_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    num_trades: int
    equity_curve: List[float]
    dates: List[str]


class BacktestResponse(BaseModel):
    results: List[BacktestResultResponse]
    buy_hold_return: float
    best_strategy: str


class StockDataResponse(BaseModel):
    ticker: str
    dates: List[str]
    close_prices: List[float]
    indicators: Dict[str, List[Optional[float]]]
