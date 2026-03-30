from abc import ABC, abstractmethod
from typing import Dict, Optional, List
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt


class BaseModel(ABC):
    """Common interface for all prediction models."""

    @abstractmethod
    def get_name(self) -> str:
        pass

    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        pass

    @abstractmethod
    def predict_future(self, last_known_data: pd.DataFrame, steps: int = 30) -> list:
        pass

    @staticmethod
    def get_tunable_params() -> Dict:
        """Return {param_name: {type, default, min, max, description}}."""
        return {}

    def get_feature_importance(self, feature_names: List[str]) -> Optional[Dict[str, float]]:
        """Return feature importances as {name: score}. None if not supported."""
        return None

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        predictions = self.predict(X_test)
        return {
            "rmse": sqrt(mean_squared_error(y_test, predictions)),
            "mae": mean_absolute_error(y_test, predictions),
            "r2": r2_score(y_test, predictions),
        }
