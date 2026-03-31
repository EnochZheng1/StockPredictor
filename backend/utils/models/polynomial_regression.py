import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from utils.models.base_model import BaseModel


class PolynomialRegressionModel(BaseModel):

    def __init__(self, degree=2):
        self.degree = degree
        self.pipeline = Pipeline([
            ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
            ("reg", LinearRegression()),
        ])

    def get_name(self) -> str:
        return f"Polynomial Regression (deg={self.degree})"

    @staticmethod
    def get_tunable_params():
        return {
            "degree": {"type": "int", "default": 2, "min": 1, "max": 4, "description": "Polynomial degree"},
        }

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.pipeline.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict(X)

    def predict_future(self, last_known_data: pd.DataFrame, steps: int = 30) -> list:
        predictions = []
        current = last_known_data.iloc[-1:].values.copy()

        for _ in range(steps):
            pred = self.pipeline.predict(np.clip(current, -1e6, 1e6))[0]
            pred = float(np.clip(pred, -1e10, 1e10))
            predictions.append(pred)
            current = np.roll(current, -1, axis=1)
            current[0, -1] = pred

        return predictions
