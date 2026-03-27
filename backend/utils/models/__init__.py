from utils.models.arima_model import ARIMAModel
from utils.models.linear_regression import LinearRegressionModel
from utils.models.lstm_model import LSTMModel
from utils.models.random_forest import RandomForestModel
from utils.models.xgboost_model import XGBoostModel
from utils.models.polynomial_regression import PolynomialRegressionModel

MODEL_REGISTRY = {
    "arima": ARIMAModel,
    "linear_regression": LinearRegressionModel,
    "lstm": LSTMModel,
    "random_forest": RandomForestModel,
    "xgboost": XGBoostModel,
    "polynomial_regression": PolynomialRegressionModel,
}

# Prophet has heavy dependencies - only register if available
try:
    from utils.models.prophet_model import ProphetModel
    MODEL_REGISTRY["prophet"] = ProphetModel
except ImportError:
    pass


def get_model(name: str, **kwargs):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name](**kwargs)


def list_models():
    return list(MODEL_REGISTRY.keys())
