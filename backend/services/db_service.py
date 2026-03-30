import os
import json
import sqlite3
import logging
from datetime import datetime
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "stockpredictor.db")


def _get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = _get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS prediction_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            model_name TEXT NOT NULL,
            period TEXT NOT NULL,
            steps INTEGER NOT NULL,
            rmse REAL,
            mae REAL,
            r2 REAL,
            params TEXT,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()
    logger.info("Database initialized at %s", DB_PATH)


def save_prediction(ticker: str, model_name: str, period: str, steps: int,
                    metrics: Dict[str, float], params: Optional[Dict] = None):
    conn = _get_conn()
    conn.execute(
        """INSERT INTO prediction_history (ticker, model_name, period, steps, rmse, mae, r2, params, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (ticker, model_name, period, steps,
         metrics.get("rmse"), metrics.get("mae"), metrics.get("r2"),
         json.dumps(params) if params else None,
         datetime.utcnow().isoformat()),
    )
    conn.commit()
    conn.close()


def get_history(ticker: Optional[str] = None, limit: int = 50) -> List[Dict]:
    conn = _get_conn()
    if ticker:
        rows = conn.execute(
            "SELECT * FROM prediction_history WHERE ticker = ? ORDER BY created_at DESC LIMIT ?",
            (ticker.upper(), limit),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM prediction_history ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# Initialize on import
init_db()
