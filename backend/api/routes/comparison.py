from fastapi import APIRouter, HTTPException

from api.schemas import ComparisonRequest, ComparisonResponse, PredictionResponse, MetricsResponse
from services.comparison_service import run_comparison

router = APIRouter()


@router.post("/compare", response_model=ComparisonResponse)
def compare(request: ComparisonRequest):
    try:
        result = run_comparison(
            ticker=request.ticker,
            model_names=request.model_names,
            steps=request.steps,
        )
        responses = []
        for r in result.results:
            responses.append(PredictionResponse(
                model_name=r.model_name,
                metrics=MetricsResponse(**r.metrics),
                test_predictions=r.test_predictions,
                test_dates=r.test_dates,
                future_predictions=r.future_predictions,
                future_dates=r.future_dates,
            ))
        return ComparisonResponse(
            results=responses,
            summary=result.summary,
            best_model=result.best_model,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
