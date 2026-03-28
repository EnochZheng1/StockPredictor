import os
import time
import logging
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from api.routes import stocks, predictions, comparison

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


@app.get("/")
def root():
    return {"message": "StockPredictor API", "docs": "/docs"}
