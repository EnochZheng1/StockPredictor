import logging
from typing import List
import numpy as np
from math import sqrt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit

from services.prediction_service import PredictionResult
from services.data_service import DataBundle
from utils.models import get_model

logger = logging.getLogger(__name__)


def _compute_metrics(y_true, y_pred):
    return {
        "rmse": sqrt(mean_squared_error(y_true, y_pred)),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }


def simple_average_ensemble(
    base_results: List[PredictionResult],
    data: DataBundle,
    steps: int,
    test_date_strs: List[str],
    future_date_strs: List[str],
) -> PredictionResult:
    test_matrix = np.array([r.test_predictions for r in base_results])
    future_matrix = np.array([r.future_predictions for r in base_results])

    avg_test = test_matrix.mean(axis=0)
    avg_future = future_matrix.mean(axis=0)

    metrics = _compute_metrics(data.y_test.values, avg_test)

    return PredictionResult(
        model_name="Ensemble (Average)",
        metrics=metrics,
        test_predictions=avg_test.tolist(),
        test_dates=test_date_strs,
        future_predictions=avg_future.tolist(),
        future_dates=future_date_strs,
    )


def weighted_average_ensemble(
    base_results: List[PredictionResult],
    data: DataBundle,
    steps: int,
    test_date_strs: List[str],
    future_date_strs: List[str],
) -> PredictionResult:
    # Weight by inverse RMSE (lower RMSE = higher weight)
    rmses = np.array([r.metrics["rmse"] for r in base_results])
    weights = 1.0 / (rmses + 1e-10)  # prevent division by zero
    weights = weights / weights.sum()

    test_matrix = np.array([r.test_predictions for r in base_results])
    future_matrix = np.array([r.future_predictions for r in base_results])

    weighted_test = np.average(test_matrix, axis=0, weights=weights)
    weighted_future = np.average(future_matrix, axis=0, weights=weights)

    metrics = _compute_metrics(data.y_test.values, weighted_test)

    return PredictionResult(
        model_name="Ensemble (Weighted)",
        metrics=metrics,
        test_predictions=weighted_test.tolist(),
        test_dates=test_date_strs,
        future_predictions=weighted_future.tolist(),
        future_dates=future_date_strs,
    )


def stacking_ensemble(
    base_results: List[PredictionResult],
    data: DataBundle,
    steps: int,
    test_date_strs: List[str],
    future_date_strs: List[str],
) -> PredictionResult:
    n_models = len(base_results)
    n_train = len(data.X_train)

    # Generate out-of-fold predictions using TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=5)
    oof_predictions = np.full((n_train, n_models), np.nan)
    valid_models = list(range(n_models))

    for fold_train_idx, fold_val_idx in tscv.split(data.X_train):
        X_fold_train = data.X_train.iloc[fold_train_idx]
        y_fold_train = data.y_train.iloc[fold_train_idx]
        X_fold_val = data.X_train.iloc[fold_val_idx]

        for j in list(valid_models):
            try:
                model = get_model(base_results[j].model_key)
                model.train(X_fold_train, y_fold_train)
                fold_preds = model.predict(X_fold_val)
                oof_predictions[fold_val_idx, j] = fold_preds
            except Exception:
                valid_models = [m for m in valid_models if m != j]

    if len(valid_models) < 2:
        raise ValueError("Stacking needs at least 2 models that succeed across all folds")

    # Keep only valid model columns
    oof_predictions = oof_predictions[:, valid_models]

    # Use rows that have predictions (skip first fold's training rows)
    valid_rows = ~np.isnan(oof_predictions).any(axis=1)
    meta_X = oof_predictions[valid_rows]
    meta_y = data.y_train.values[valid_rows]

    # Train meta-learner
    meta_model = Ridge(alpha=1.0)
    meta_model.fit(meta_X, meta_y)

    # Test predictions: use base models' test predictions as meta-features
    test_meta_X = np.column_stack([
        base_results[j].test_predictions for j in valid_models
    ])
    stacked_test = meta_model.predict(test_meta_X)

    # Future predictions
    future_meta_X = np.column_stack([
        base_results[j].future_predictions for j in valid_models
    ])
    stacked_future = meta_model.predict(future_meta_X)

    metrics = _compute_metrics(data.y_test.values, stacked_test)

    return PredictionResult(
        model_name="Ensemble (Stacking)",
        metrics=metrics,
        test_predictions=stacked_test.tolist(),
        test_dates=test_date_strs,
        future_predictions=stacked_future.tolist(),
        future_dates=future_date_strs,
    )


ENSEMBLE_REGISTRY = {
    "ensemble_average": simple_average_ensemble,
    "ensemble_weighted": weighted_average_ensemble,
    "ensemble_stacking": stacking_ensemble,
}


def list_ensemble_methods():
    return list(ENSEMBLE_REGISTRY.keys())


def run_ensembles(
    ensemble_methods: List[str],
    base_results: List[PredictionResult],
    data: DataBundle,
    steps: int,
    test_date_strs: List[str],
    future_date_strs: List[str],
) -> List[PredictionResult]:
    if len(base_results) < 2:
        return []

    results = []
    for method_name in ensemble_methods:
        if method_name not in ENSEMBLE_REGISTRY:
            continue
        try:
            fn = ENSEMBLE_REGISTRY[method_name]
            result = fn(base_results, data, steps, test_date_strs, future_date_strs)
            logger.info("Ensemble %s: RMSE=%.4f", method_name, result.metrics["rmse"])
            results.append(result)
        except Exception as e:
            logger.error("Ensemble %s failed: %s", method_name, e)
            continue

    return results
