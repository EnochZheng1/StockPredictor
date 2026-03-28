from typing import List, Dict, Optional
from dataclasses import dataclass
import pandas as pd

from utils.models import get_model
from services.data_service import prepare_data
from services.prediction_service import PredictionResult
from services.ensemble_service import run_ensembles


@dataclass
class ComparisonResult:
    results: List[PredictionResult]
    summary: List[Dict[str, float]]
    best_model: str


def run_comparison(
    ticker: str,
    model_names: List[str],
    steps: int = 30,
    ensemble_methods: Optional[List[str]] = None,
) -> ComparisonResult:
    data = prepare_data(ticker)
    results = []

    # Generate future dates once
    last_date = pd.to_datetime(data.test_dates.iloc[-1])
    future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=steps)
    future_date_strs = [str(d.date()) for d in future_dates]
    test_date_strs = [str(d.date()) for d in pd.to_datetime(data.test_dates)]

    for name in model_names:
        try:
            model = get_model(name)
            model.train(data.X_train, data.y_train)
            metrics = model.evaluate(data.X_test, data.y_test)
            test_preds = model.predict(data.X_test)
            future_preds = model.predict_future(data.X_test, steps=steps)

            results.append(PredictionResult(
                model_name=model.get_name(),
                metrics=metrics,
                test_predictions=[float(p) for p in test_preds],
                test_dates=test_date_strs,
                future_predictions=[float(p) for p in future_preds],
                future_dates=future_date_strs,
                model_key=name,
            ))
        except Exception:
            continue

    # Run ensemble methods if requested
    if ensemble_methods and len(results) >= 2:
        ensemble_results = run_ensembles(
            ensemble_methods=ensemble_methods,
            base_results=results,
            data=data,
            steps=steps,
            test_date_strs=test_date_strs,
            future_date_strs=future_date_strs,
        )
        results.extend(ensemble_results)

    # Build summary table sorted by RMSE
    summary = []
    for r in results:
        summary.append({
            "model_name": r.model_name,
            "rmse": r.metrics["rmse"],
            "mae": r.metrics["mae"],
            "r2": r.metrics["r2"],
        })
    summary.sort(key=lambda x: x["rmse"])

    best_model = summary[0]["model_name"] if summary else "N/A"

    return ComparisonResult(
        results=results,
        summary=summary,
        best_model=best_model,
    )
