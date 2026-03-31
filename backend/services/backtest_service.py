import logging
import time
from typing import List, Dict, Optional
from dataclasses import dataclass
import numpy as np
import pandas as pd

from utils.models import get_model
from services.data_service import prepare_data

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    model_name: str
    total_return: float       # percentage
    buy_hold_return: float    # percentage
    sharpe_ratio: float
    max_drawdown: float       # percentage
    win_rate: float           # percentage
    num_trades: int
    equity_curve: List[float]
    dates: List[str]


@dataclass
class BacktestResponse:
    results: List[BacktestResult]
    buy_hold_return: float
    best_strategy: str


def _compute_backtest(predictions: np.ndarray, actual_prices: np.ndarray, dates: List[str]) -> BacktestResult:
    """Run a simple long/flat strategy: buy when predicted next price > current price."""
    n = len(actual_prices)
    daily_returns = np.diff(actual_prices) / np.where(actual_prices[:-1] == 0, 1e-10, actual_prices[:-1])

    # Strategy: position[i] = 1 if predicted[i+1] > predicted[i], else 0 (flat)
    positions = np.zeros(n - 1)
    for i in range(n - 1):
        if i < len(predictions) - 1 and predictions[i + 1] > predictions[i]:
            positions[i] = 1  # long

    strategy_returns = positions * daily_returns
    buy_hold_returns = daily_returns

    # Equity curve
    equity = [1.0]
    for r in strategy_returns:
        equity.append(equity[-1] * (1 + r))

    # Metrics
    total_return = (equity[-1] - 1) * 100
    buy_hold_return = (np.prod(1 + buy_hold_returns) - 1) * 100

    # Sharpe ratio (annualized, 252 trading days)
    if np.std(strategy_returns) > 0:
        sharpe = np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252)
    else:
        sharpe = 0.0

    # Max drawdown
    peak = np.maximum.accumulate(equity)
    drawdown = (np.array(equity) - peak) / peak
    max_dd = abs(float(np.min(drawdown))) * 100

    # Win rate
    trades = strategy_returns[positions == 1]
    num_trades = len(trades)
    win_rate = (float(np.sum(trades > 0)) / num_trades * 100) if num_trades > 0 else 0.0

    return BacktestResult(
        model_name="",
        total_return=round(total_return, 2),
        buy_hold_return=round(buy_hold_return, 2),
        sharpe_ratio=round(float(sharpe), 2),
        max_drawdown=round(max_dd, 2),
        win_rate=round(win_rate, 2),
        num_trades=num_trades,
        equity_curve=[round(v, 4) for v in equity],
        dates=dates[:len(equity)],
    )


def run_backtest(
    ticker: str,
    model_names: List[str],
    period: str = "5y",
    model_params: Optional[Dict[str, Dict]] = None,
) -> BacktestResponse:
    logger.info("Starting backtest for %s with models: %s", ticker, model_names)
    t0 = time.time()

    data = prepare_data(ticker, period=period)
    actual_test = data.y_test.values
    test_dates = [str(d.date()) for d in pd.to_datetime(data.test_dates)]

    results = []
    buy_hold = (actual_test[-1] / actual_test[0] - 1) * 100

    for name in model_names:
        try:
            kwargs = (model_params or {}).get(name, {})
            model = get_model(name, **kwargs)
            model.train(data.X_train, data.y_train)
            preds = model.predict(data.X_test)

            bt = _compute_backtest(preds, actual_test, test_dates)
            bt.model_name = model.get_name()
            results.append(bt)
            logger.info("Backtest %s: return=%.2f%%, sharpe=%.2f", name, bt.total_return, bt.sharpe_ratio)
        except Exception as e:
            logger.error("Backtest %s failed: %s", name, e)

    results.sort(key=lambda r: r.total_return, reverse=True)
    best = results[0].model_name if results else "N/A"

    logger.info("Backtest complete in %.2fs", time.time() - t0)
    return BacktestResponse(results=results, buy_hold_return=round(buy_hold, 2), best_strategy=best)
