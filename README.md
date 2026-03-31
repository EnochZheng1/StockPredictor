# StockPredictor

A multi-model stock price prediction platform. Compare 7 prediction models + 3 ensemble methods side-by-side, run backtests, and visualize results through an interactive React dashboard.

## Features

- **7 Prediction Models**: ARIMA, Linear Regression, LSTM, Random Forest, XGBoost, Prophet, Polynomial Regression
- **3 Ensemble Methods**: Simple Average, Weighted Average (inverse RMSE), Stacking (Ridge)
- **15+ Technical Indicators**: SMA, EMA, RSI, MACD, Bollinger Bands, ADX, OBV, ATR, Ichimoku, and more
- **Backtesting Engine**: Simulate buy/sell strategies with P&L tracking, Sharpe ratio, max drawdown
- **Hyperparameter Tuning**: Adjust model parameters directly from the UI
- **Feature Importance**: Visualize which indicators drive predictions (Random Forest, XGBoost)
- **CSV Export**: Download metrics and prediction data
- **Dark Mode**: Toggle between light and dark themes
- **Responsive Design**: Works on desktop, tablet, and mobile

## Models

| Model | Type | Library |
|-------|------|---------|
| ARIMA | Time Series | statsmodels |
| Linear Regression | Feature-based | scikit-learn |
| LSTM | Deep Learning | PyTorch |
| Random Forest | Ensemble | scikit-learn |
| XGBoost | Gradient Boosting | xgboost |
| Prophet | Time Series | prophet |
| Polynomial Regression | Feature-based | scikit-learn |

## Quick Start

### Option 1: Docker (recommended)

```bash
docker-compose up --build
```

Open `http://localhost:4290`.

### Option 2: Local Development

#### Prerequisites

- Python 3.9+
- Node.js 18+
- [TA-Lib C library](https://github.com/cgohlke/talib-build/) (optional, for SAR/ADX/Stochastic)

#### Installation

```bash
git clone https://github.com/EnochZheng1/StockPredictor.git
cd StockPredictor
npm install                    # root dependencies (concurrently)
pip install -r requirements.txt  # Python dependencies
npm run install:frontend       # React dependencies
cp .env.example .env           # configure ports (optional)
```

#### Run

```bash
npm start
```

This launches:
- **Backend API** at `http://localhost:4289` (FastAPI + uvicorn)
- **Frontend UI** at `http://localhost:4290` (Vite + React)

### Run tests

```bash
npm test                  # all tests
npm run test:backend      # pytest (33 tests)
npm run test:frontend     # vitest (14 tests)
```

## How to Use

1. Enter a stock ticker (e.g., `AAPL`) and select a time period, then click **Fetch Data**
2. Select models using checkboxes -- click **Tune** to adjust hyperparameters
3. Optionally select ensemble methods (requires 2+ models)
4. Set forecast days and click **Run Comparison**
5. View prediction chart, metrics table, and feature importance
6. Click **Run Backtest** to simulate a long/flat trading strategy
7. Click **Export Metrics CSV** or **Export Predictions CSV** to download results

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/stocks/{ticker}` | Historical data + technical indicators |
| GET | `/api/models` | Available models, ensemble methods, tunable params |
| GET | `/api/sentiment/{ticker}` | News sentiment analysis (VADER) |
| GET | `/api/history` | Prediction history (optional `?ticker=` filter) |
| POST | `/api/predict` | Single model prediction |
| POST | `/api/compare` | Multi-model comparison with optional ensembles |
| POST | `/api/backtest` | Run backtest on selected models |
| POST | `/api/portfolio` | Compare predictions across multiple tickers |
| WS | `/ws/price/{ticker}` | Live price updates (WebSocket, 10s interval) |

Interactive API docs: `http://localhost:4289/docs`

#### Example: Compare models

```bash
curl -X POST http://localhost:4289/api/compare \
  -H "Content-Type: application/json" \
  -d '{
    "ticker": "AAPL",
    "model_names": ["random_forest", "xgboost", "arima"],
    "steps": 30,
    "period": "5y",
    "ensemble_methods": ["ensemble_weighted"]
  }'
```

## Project Structure

```
StockPredictor/
├── docker-compose.yml        # Docker orchestration
├── package.json              # Root scripts (npm start, npm test)
├── requirements.txt          # Python dependencies
├── .env.example              # Port/URL configuration template
├── .github/workflows/ci.yml  # GitHub Actions CI
├── backend/
│   ├── Dockerfile
│   ├── app.py                # FastAPI entry point
│   ├── api/
│   │   ├── schemas.py        # Pydantic models
│   │   └── routes/           # stocks, predictions, comparison, backtest
│   ├── services/
│   │   ├── data_service.py   # Data fetching + train/test split
│   │   ├── prediction_service.py
│   │   ├── comparison_service.py
│   │   ├── ensemble_service.py
│   │   └── backtest_service.py
│   ├── tests/                # pytest test suite
│   └── utils/
│       ├── data_fetcher.py   # yfinance wrapper with caching
│       ├── data_analysis.py  # 15+ technical indicators
│       └── models/           # 7 prediction models + registry
└── frontend/
    ├── Dockerfile
    ├── nginx.conf
    └── src/
        ├── api/stockApi.js
        ├── components/       # TickerInput, ModelSelector, Charts, Tables
        └── pages/Dashboard.jsx
```

## Evaluation Metrics

| Metric | Description | Goal |
|--------|-------------|------|
| RMSE | Root Mean Squared Error | Lower is better |
| MAE | Mean Absolute Error | Lower is better |
| R2 | R-squared | Closer to 1 is better |

Data is split 80/20 using a **time-ordered split** (no shuffle) to prevent data leakage.

## Backtesting

The backtesting engine runs a **long/flat strategy** on the test period:
- **Buy** when the model predicts the next price will be higher
- **Hold cash** when the model predicts the price will drop

Metrics: Total Return, Sharpe Ratio, Max Drawdown, Win Rate, number of trades.

## Ports

| Service | Port |
|---------|------|
| Backend API | 4289 |
| Frontend UI | 4290 |
