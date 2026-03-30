import os
import time
import json
import asyncio
import logging
from dotenv import load_dotenv
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from api.routes import stocks, predictions, comparison, backtest, portfolio

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:4290")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="StockPredictor API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0 = time.time()
    response = await call_next(request)
    elapsed = time.time() - t0
    logger.info("%s %s %d (%.2fs)", request.method, request.url.path, response.status_code, elapsed)
    return response


app.include_router(stocks.router, prefix="/api")
app.include_router(predictions.router, prefix="/api")
app.include_router(comparison.router, prefix="/api")
app.include_router(backtest.router, prefix="/api")
app.include_router(portfolio.router, prefix="/api")


@app.get("/")
def root():
    return {"message": "StockPredictor API", "docs": "/docs"}


@app.websocket("/ws/price/{ticker}")
async def websocket_price(websocket: WebSocket, ticker: str):
    """Stream live price updates for a ticker every 10 seconds."""
    import yfinance as yf

    await websocket.accept()
    logger.info("WebSocket connected for %s", ticker)
    try:
        while True:
            try:
                stock = yf.Ticker(ticker)
                info = stock.fast_info
                data = {
                    "ticker": ticker.upper(),
                    "price": round(float(info.last_price), 2) if hasattr(info, "last_price") else None,
                    "previous_close": round(float(info.previous_close), 2) if hasattr(info, "previous_close") else None,
                    "timestamp": time.time(),
                }
                if data["price"] and data["previous_close"]:
                    data["change"] = round(data["price"] - data["previous_close"], 2)
                    data["change_pct"] = round((data["change"] / data["previous_close"]) * 100, 2)
                await websocket.send_text(json.dumps(data))
            except Exception as e:
                await websocket.send_text(json.dumps({"error": str(e)}))
            await asyncio.sleep(10)
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected for %s", ticker)
