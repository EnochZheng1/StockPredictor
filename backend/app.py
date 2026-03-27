from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import stocks, predictions, comparison

app = FastAPI(title="StockPredictor API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:4290",  # Vite dev server
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(stocks.router, prefix="/api")
app.include_router(predictions.router, prefix="/api")
app.include_router(comparison.router, prefix="/api")


@app.get("/")
def root():
    return {"message": "StockPredictor API", "docs": "/docs"}
