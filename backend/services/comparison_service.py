import logging
import time
from typing import List, Dict, Optional
from dataclasses import dataclass
import pandas as pd

from utils.models import get_model
from services.data_service import prepare_data
from services.prediction_service import PredictionResult
from services.ensemble_service import run_ensembles

logger = logging.getLogger(__name__)


@dataclass
class ComparisonResult:
    results: List[PredictionResult]
    summary: List[Dict[str, float]]
    best_model: str
    errors: List[str] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []


def run_comparison(
    ticker: str,
    model_names: List[str],
    steps: int = 30,
    period: str = "5y",
    ensemble_methods: Optional[List[str]] = None,
) -> ComparisonResult:
    logger.info("Starting comparison for %s with models: %s", ticker, model_names)
    t0 = time.time()

    data = prepare_data(ticker, period=period)
    logger.info("Data prepared: %d train, %d test samples", len(data.X_train), len(data.X_test))

    results = []
    errors = []

    # Generate future dates once
    last_date = pd.to_datetime(data.test_dates.iloc[-1])
    future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=steps)
    future_date_strs = [str(d.date()) for d in future_dates]
    test_date_strs = [str(d.date()) for d in pd.to_datetime(data.test_dates)]

    for name in model_names:
        try:
            model_t0 = time.time()
            model = get_model(name)
            model.train(data.X_train, data.y_train)
            metrics = model.evaluate(data.X_test, data.y_test)
            test_preds = model.predict(data.X_test)
            future_preds = model.predict_future(data.X_test, steps=steps)
            feat_imp = model.get_feature_importance(data.feature_columns)

            elapsed = time.time() - model_t0
            logger.info("Model %s completed in %.2fs (RMSE=%.4f)", name, elapsed, metrics["rmse"])

            results.append(PredictionResult(
                model_name=model.get_name(),
                metrics=metrics,
                test_predictions=[float(p) for p in test_preds],
                test_dates=test_date_strs,
                future_predictions=[float(p) for p in future_preds],
                future_dates=future_date_strs,
                model_key=name,
                feature_importance=feat_imp,
            ))
        except Exception as e:
            logger.error("Model %s failed: %s", name, e)
            errors.append(f"{name}: {str(e)}")
            continue

    # Run ensemble methods if requested
    if ensemble_methods and len(results) >= 2:
        logger.info("Running ensemble methods: %s", ensemble_methods)
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
    logger.info("Comparison complete in %.2fs. Best: %s", time.time() - t0, best_model)

    return ComparisonResult(
        results=results,
        summary=summary,
        best_model=best_model,
        errors=errors,
    )
