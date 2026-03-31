from fastapi import APIRouter, HTTPException

from api.schemas import PredictionRequest, PredictionResponse, MetricsResponse
from services.prediction_service import run_prediction

router = APIRouter()


@router.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        result = run_prediction(
            ticker=request.ticker,
            model_name=request.model_name,
            steps=request.steps,
            period=request.period,
        )
        return PredictionResponse(
            model_name=result.model_name,
            metrics=MetricsResponse(**result.metrics),
            test_predictions=result.test_predictions,
            test_dates=result.test_dates,
            future_predictions=result.future_predictions,
            future_dates=result.future_dates,
            feature_importance=result.feature_importance,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
