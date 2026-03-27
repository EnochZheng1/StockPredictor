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

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.pipeline.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict(X)

    def predict_future(self, last_known_data: pd.DataFrame, steps: int = 30) -> list:
        predictions = []
        current = last_known_data.iloc[-1:].values.copy()

        for _ in range(steps):
            pred = self.pipeline.predict(current)[0]
            predictions.append(float(pred))

        return predictions
