# StockPredictor

A multi-model stock price prediction platform. Compare 7 different math/stats models side-by-side to forecast stock prices using historical data and technical indicators.

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

## Technical Indicators

The data pipeline computes 15+ indicators automatically: SMA, EMA, RSI, MACD, Bollinger Bands, Parabolic SAR, ADX, Stochastic Oscillator, OBV, Momentum, ATR, Ichimoku Cloud, and Williams %R.

## Prerequisites

- Python 3.9+
- Node.js 18+
- [TA-Lib C library](https://github.com/cgohlke/talib-build/) (required for some technical indicators)

## Installation

### 1. Clone and install root dependencies

```bash
git clone <repo-url>
cd StockPredictor
npm install
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Install frontend dependencies

```bash
npm run install:frontend
```

## Usage

### Start both servers with one command

```bash
npm start
```

This launches:
- **Backend API** at `http://localhost:4289` (FastAPI + uvicorn)
- **Frontend UI** at `http://localhost:4290` (Vite + React)

Open `http://localhost:4290` in your browser.

### Start servers individually

```bash
# Backend only
npm run start:backend

# Frontend only
npm run start:frontend
```

### How to use

1. Enter a stock ticker symbol (e.g., `AAPL`, `MSFT`, `GOOGL`) and click **Fetch Data**
2. Select one or more prediction models using the checkboxes
3. Set the number of forecast days (default: 30)
4. Click **Run Comparison**
5. View the prediction chart and metrics comparison table

### API endpoints

The backend API is also available directly:

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/stocks/{ticker}` | Fetch historical data + indicators |
| GET | `/api/models` | List available models |
| POST | `/api/predict` | Run a single model prediction |
| POST | `/api/compare` | Compare multiple models |

Interactive API docs are available at `http://localhost:4289/docs`.

#### Example: Compare models via API

```bash
curl -X POST http://localhost:4289/api/compare \
  -H "Content-Type: application/json" \
  -d '{"ticker": "AAPL", "model_names": ["random_forest", "xgboost", "arima"], "steps": 30}'
```

## Project Structure

```
StockPredictor/
├── package.json              # Root scripts (npm start runs both servers)
├── requirements.txt          # Python dependencies
├── backend/
│   ├── app.py                # FastAPI entry point
│   ├── api/
│   │   ├── schemas.py        # Pydantic request/response models
│   │   └── routes/           # API route handlers
│   ├── services/
│   │   ├── data_service.py   # Data fetching + train/test splitting
│   │   ├── prediction_service.py
│   │   └── comparison_service.py
│   └── utils/
│       ├── data_fetcher.py   # yfinance wrapper
│       ├── data_analysis.py  # Technical indicators
│       └── models/           # All 7 prediction models
├── frontend/
│   └── src/
│       ├── api/stockApi.js   # Backend API client
│       ├── components/       # React components
│       └── pages/Dashboard.jsx
```

## Evaluation Metrics

Models are compared using:
- **RMSE** (Root Mean Squared Error) - lower is better
- **MAE** (Mean Absolute Error) - lower is better
- **R2** (R-squared) - closer to 1 is better

Data is split 80/20 using a **time-ordered split** (no shuffle) to prevent data leakage.

## Ports

| Service | Port |
|---------|------|
| Backend API | 4289 |
| Frontend UI | 4290 |
