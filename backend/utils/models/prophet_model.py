import pandas as pd
import numpy as np
import logging
from utils.models.base_model import BaseModel

# Suppress Prophet's verbose logging
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)

try:
    from prophet import Prophet

    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False


class ProphetModel(BaseModel):

    def __init__(self):
        if not PROPHET_AVAILABLE:
            raise ImportError("prophet is not installed. Install with: pip install prophet")
        self.model = None
        self._train_dates = None

    def get_name(self) -> str:
        return "Prophet"

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        # Prophet needs a DataFrame with 'ds' (date) and 'y' (value) columns
        if "Date" in X.columns:
            dates = pd.to_datetime(X["Date"])
        elif hasattr(X, "index") and hasattr(X.index, "date"):
            dates = pd.to_datetime(X.index)
        else:
            dates = pd.RangeIndex(len(y))

        df = pd.DataFrame({"ds": dates, "y": y.values})
        self._train_dates = dates
        self.model = Prophet(daily_seasonality=True, yearly_seasonality=True)
        self.model.fit(df)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if "Date" in X.columns:
            dates = pd.to_datetime(X["Date"])
        elif hasattr(X, "index") and hasattr(X.index, "date"):
            dates = pd.to_datetime(X.index)
        else:
            # Generate dates continuing from training
            last_date = self._train_dates.iloc[-1]
            dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=len(X), freq="B")

        future = pd.DataFrame({"ds": dates})
        forecast = self.model.predict(future)
        return forecast["yhat"].values

    def predict_future(self, last_known_data: pd.DataFrame, steps: int = 30) -> list:
        future = self.model.make_future_dataframe(periods=steps, freq="B")
        forecast = self.model.predict(future)
        return forecast["yhat"].iloc[-steps:].tolist()
