import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from utils.models.base_model import BaseModel


class XGBoostModel(BaseModel):

    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42):
        self.model = XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state,
            verbosity=0,
        )

    def get_name(self) -> str:
        return "XGBoost"

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

        return predictions
