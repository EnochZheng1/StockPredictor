import services.db_service as db


class TestInitDb:
    def test_creates_table(self):
        # Query sqlite_master to verify prediction_history exists
        conn = db._get_conn()
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        conn.close()
        assert any(r["name"] == "prediction_history" for r in tables)


class TestSavePrediction:
    def test_inserts_row(self):
        db.save_prediction(
            "AAPL", "Linear Regression", "5y", 30,
            {"rmse": 1.5, "mae": 1.0, "r2": 0.9},
        )
        history = db.get_history()
        assert len(history) == 1

    def test_stores_metrics(self):
        db.save_prediction(
            "AAPL", "RF", "5y", 30,
            {"rmse": 2.0, "mae": 1.5, "r2": 0.85},
        )
        row = db.get_history()[0]
        assert row["rmse"] == 2.0
        assert row["mae"] == 1.5
        assert row["r2"] == 0.85

    def test_stores_params_as_json(self):
        db.save_prediction(
            "AAPL", "RF", "5y", 30,
            {"rmse": 1.0, "mae": 1.0, "r2": 0.9},
            params={"n_estimators": 100},
        )
        row = db.get_history()[0]
        assert "100" in row["params"]

    def test_null_params(self):
        db.save_prediction(
            "AAPL", "RF", "5y", 30,
            {"rmse": 1.0, "mae": 1.0, "r2": 0.9},
            params=None,
        )
        row = db.get_history()[0]
        assert row["params"] is None


class TestGetHistory:
    def test_returns_all(self):
        for i in range(3):
            db.save_prediction(
                "AAPL", f"Model{i}", "5y", 30,
                {"rmse": float(i), "mae": 0, "r2": 0},
            )
        assert len(db.get_history()) == 3

    def test_filters_by_ticker(self):
        db.save_prediction(
            "AAPL", "RF", "5y", 30, {"rmse": 1, "mae": 1, "r2": 0.9}
        )
        db.save_prediction(
            "MSFT", "RF", "5y", 30, {"rmse": 2, "mae": 2, "r2": 0.8}
        )
        history = db.get_history(ticker="AAPL")
        assert len(history) == 1
        assert history[0]["ticker"] == "AAPL"

    def test_respects_limit(self):
        for i in range(10):
            db.save_prediction(
                "AAPL", f"M{i}", "5y", 30,
                {"rmse": 0, "mae": 0, "r2": 0},
            )
        assert len(db.get_history(limit=3)) == 3
