import pandas as pd
import numpy as np
from typing import Dict, Optional, List
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

    @staticmethod
    def get_tunable_params():
        return {
            "n_estimators": {"type": "int", "default": 100, "min": 10, "max": 500, "description": "Number of boosting rounds"},
            "learning_rate": {"type": "float", "default": 0.1, "min": 0.01, "max": 1.0, "description": "Learning rate"},
            "max_depth": {"type": "int", "default": 6, "min": 1, "max": 15, "description": "Max tree depth"},
        }

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def get_feature_importance(self, feature_names: List[str]) -> Optional[Dict[str, float]]:
        importances = self.model.feature_importances_
        return dict(zip(feature_names, [float(v) for v in importances]))

    def predict_future(self, last_known_data: pd.DataFrame, steps: int = 30) -> list:
        predictions = []
        current = last_known_data.iloc[-1:].values.copy()

        for _ in range(steps):
            pred = self.model.predict(current)[0]
            predictions.append(float(pred))
            current = np.roll(current, -1, axis=1)
            current[0, -1] = pred

        return predictions
