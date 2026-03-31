import pytest
import pandas as pd
import numpy as np


@pytest.mark.slow
class TestLSTMModel:
    """LSTM tests use small data and few epochs for speed."""

    @pytest.fixture
    def small_data(self):
        np.random.seed(42)
        n = 60
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        df = pd.DataFrame({
            "f1": close + np.random.randn(n) * 0.3,
            "f2": np.random.randn(n) * 10 + 50,
            "f3": np.random.randn(n) * 0.5,
        })
        y = pd.Series(close, name="Close")
        split = 45
        return {
            "X_train": df.iloc[:split],
            "X_test": df.iloc[split:],
            "y_train": y.iloc[:split],
            "y_test": y.iloc[split:],
        }

    def _make_model(self):
        from utils.models.lstm_model import LSTMModel
        return LSTMModel(hidden_layer_size=8, sequence_length=5, epochs=2, lr=0.01)

    def test_train_predict(self, small_data):
        model = self._make_model()
        model.train(small_data["X_train"], small_data["y_train"])
        preds = model.predict(small_data["X_test"])
        assert len(preds) == len(small_data["X_test"])
        assert not np.any(np.isnan(preds))

    def test_predict_future(self, small_data):
        model = self._make_model()
        model.train(small_data["X_train"], small_data["y_train"])
        future = model.predict_future(small_data["X_test"], steps=5)
        assert len(future) == 5
        assert all(isinstance(v, float) for v in future)

    def test_evaluate(self, small_data):
        model = self._make_model()
        model.train(small_data["X_train"], small_data["y_train"])
        metrics = model.evaluate(small_data["X_test"], small_data["y_test"])
        assert "rmse" in metrics
        assert "mae" in metrics
        assert "r2" in metrics

    def test_get_name(self):
        from utils.models.lstm_model import LSTMModel
        assert LSTMModel().get_name() == "LSTM"

    def test_tunable_params(self):
        from utils.models.lstm_model import LSTMModel
        params = LSTMModel.get_tunable_params()
        assert "hidden_layer_size" in params
        assert "epochs" in params
        assert "sequence_length" in params

    def test_custom_params(self, small_data):
        from utils.models.lstm_model import LSTMModel
        model = LSTMModel(hidden_layer_size=16, sequence_length=3, epochs=2)
        model.train(small_data["X_train"], small_data["y_train"])
        preds = model.predict(small_data["X_test"])
        assert len(preds) > 0
