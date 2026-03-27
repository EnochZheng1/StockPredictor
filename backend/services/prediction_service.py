from typing import Dict, List
from dataclasses import dataclass

from utils.models import get_model
from services.data_service import prepare_data


@dataclass
class PredictionResult:
    model_name: str
    metrics: Dict[str, float]
    test_predictions: List[float]
    test_dates: List[str]
    future_predictions: List[float]
    future_dates: List[str]


def run_prediction(ticker: str, model_name: str, steps: int = 30) -> PredictionResult:
    import pandas as pd

    data = prepare_data(ticker)
    model = get_model(model_name)

    model.train(data.X_train, data.y_train)
    metrics = model.evaluate(data.X_test, data.y_test)
    test_preds = model.predict(data.X_test)
    future_preds = model.predict_future(data.X_test, steps=steps)

    # Generate future dates (business days)
    last_date = pd.to_datetime(data.test_dates.iloc[-1])
    future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=steps)

    return PredictionResult(
        model_name=model.get_name(),
        metrics=metrics,
        test_predictions=[float(p) for p in test_preds],
        test_dates=[str(d.date()) for d in pd.to_datetime(data.test_dates)],
        future_predictions=[float(p) for p in future_preds],
        future_dates=[str(d.date()) for d in future_dates],
    )
