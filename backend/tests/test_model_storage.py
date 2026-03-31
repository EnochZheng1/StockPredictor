from services.model_storage import (
    save_model,
    load_model,
    _model_key,
    list_saved_models,
    clear_saved_models,
)
from utils.models.linear_regression import LinearRegressionModel
import numpy as np


class TestModelKey:
    def test_deterministic(self):
        k1 = _model_key("AAPL", "rf", "5y", {})
        k2 = _model_key("AAPL", "rf", "5y", {})
        assert k1 == k2

    def test_varies_with_params(self):
        k1 = _model_key("AAPL", "rf", "5y", {})
        k2 = _model_key("AAPL", "rf", "5y", {"n": 100})
        assert k1 != k2


class TestSaveLoadJoblib:
    def test_save_load_round_trip(self, sample_feature_data, tmp_model_dir):
        d = sample_feature_data
        model = LinearRegressionModel()
        model.train(d["X_train"], d["y_train"])
        preds_before = model.predict(d["X_test"])

        save_model(model, "AAPL", "linear_regression", "5y")
        loaded = load_model(LinearRegressionModel, "AAPL", "linear_regression", "5y")

        assert loaded is not None
        preds_after = loaded.predict(d["X_test"])
        np.testing.assert_array_almost_equal(preds_before, preds_after)

    def test_load_nonexistent_returns_none(self, tmp_model_dir):
        result = load_model(LinearRegressionModel, "ZZZZ", "fake", "1y")
        assert result is None


class TestListClear:
    def test_list_saved(self, sample_feature_data, tmp_model_dir):
        model = LinearRegressionModel()
        model.train(
            sample_feature_data["X_train"], sample_feature_data["y_train"]
        )
        save_model(model, "A", "lr", "5y")
        save_model(model, "B", "lr", "5y")
        assert len(list_saved_models()) == 2

    def test_clear(self, sample_feature_data, tmp_model_dir):
        model = LinearRegressionModel()
        model.train(
            sample_feature_data["X_train"], sample_feature_data["y_train"]
        )
        save_model(model, "A", "lr", "5y")
        count = clear_saved_models()
        assert count == 1
        assert len(list_saved_models()) == 0

    def test_save_failure_returns_false(self, tmp_model_dir, monkeypatch):
        import joblib

        monkeypatch.setattr(
            joblib,
            "dump",
            lambda *a, **kw: (_ for _ in ()).throw(OSError("fail")),
        )
        model = LinearRegressionModel()
        result = save_model(model, "X", "lr", "5y")
        assert result is False
