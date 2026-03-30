from fastapi import APIRouter, HTTPException

from api.schemas import BacktestRequest, BacktestResponse, BacktestResultResponse
from services.backtest_service import run_backtest

router = APIRouter()


@router.post("/backtest", response_model=BacktestResponse)
def backtest(request: BacktestRequest):
    try:
        result = run_backtest(
            ticker=request.ticker,
            model_names=request.model_names,
            period=request.period,
            model_params=request.model_params,
        )
        return BacktestResponse(
            results=[BacktestResultResponse(**vars(r)) for r in result.results],
            buy_hold_return=result.buy_hold_return,
            best_strategy=result.best_strategy,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
