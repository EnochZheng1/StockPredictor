import pandas as pd
import numpy as np
from typing import Dict, Optional, List
from sklearn.ensemble import RandomForestRegressor
from utils.models.base_model import BaseModel


class RandomForestModel(BaseModel):

    def __init__(self, n_estimators=100, random_state=42):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
        )

    def get_name(self) -> str:
        return "Random Forest"

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

        return predictions
