import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from utils.models.base_model import BaseModel


class ARIMAModel(BaseModel):

    def __init__(self, p=5, d=1, q=0):
        self.order = (p, d, q)
        self.model_fit = None
        self.train_series = None

    def get_name(self) -> str:
        return "ARIMA"

    @staticmethod
    def get_tunable_params():
        return {
            "p": {"type": "int", "default": 5, "min": 0, "max": 10, "description": "Autoregressive order"},
            "d": {"type": "int", "default": 1, "min": 0, "max": 3, "description": "Differencing order"},
            "q": {"type": "int", "default": 0, "min": 0, "max": 10, "description": "Moving average order"},
        }

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.train_series = y.values
        model = ARIMA(self.train_series, order=self.order)
        self.model_fit = model.fit()

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        n = len(X)
        start = len(self.train_series)
        end = start + n - 1
        predictions = self.model_fit.predict(start=start, end=end)
        return np.array(predictions)

    def predict_future(self, last_known_data: pd.DataFrame, steps: int = 30) -> list:
        forecast = self.model_fit.forecast(steps=steps)
        return forecast.tolist()
