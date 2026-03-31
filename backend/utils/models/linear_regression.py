import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from utils.models.base_model import BaseModel


class LinearRegressionModel(BaseModel):

    def __init__(self):
        self.model = LinearRegression()

    def get_name(self) -> str:
        return "Linear Regression"

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def predict_future(self, last_known_data: pd.DataFrame, steps: int = 30) -> list:
        predictions = []
        current = last_known_data.iloc[-1:].values.copy()

        for _ in range(steps):
            pred = self.model.predict(current)[0]
            predictions.append(float(pred))
            # Feed prediction back as first feature and shift others
            current = np.roll(current, -1, axis=1)
            current[0, -1] = pred

        return predictions
