import { useState } from "react";
import { runPortfolio } from "../api/stockApi";

const MODEL_LABELS = {
  arima: "ARIMA",
  linear_regression: "Linear Regression",
  lstm: "LSTM",
  random_forest: "Random Forest",
  xgboost: "XGBoost",
  prophet: "Prophet",
  polynomial_regression: "Polynomial Regression",
};

export default function PortfolioView({ models }) {
  const [tickersInput, setTickersInput] = useState("AAPL, MSFT, GOOGL, AMZN, TSLA");
  const [model, setModel] = useState("random_forest");
  const [steps, setSteps] = useState(30);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleRun = async () => {
    const tickers = tickersInput.split(",").map((t) => t.trim()).filter(Boolean);
    if (tickers.length === 0) return;
    setLoading(true);
    setError(null);
    try {
      const data = await runPortfolio(tickers, model, steps);
      setResults(data);
    } catch (err) {
      setError(err.response?.data?.detail || "Failed to run portfolio comparison");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="portfolio-view">
      <h3>Portfolio Comparison</h3>
      <p className="portfolio-desc">Compare predicted returns across multiple stocks using a single model.</p>
      <div className="portfolio-controls">
        <input
          type="text"
          value={tickersInput}
          onChange={(e) => setTickersInput(e.target.value)}
          placeholder="AAPL, MSFT, GOOGL..."
          disabled={loading}
        />
        <select value={model} onChange={(e) => setModel(e.target.value)} disabled={loading}>
          {models.map((m) => (
            <option key={m} value={m}>{MODEL_LABELS[m] || m}</option>
          ))}
        </select>
        <input
          type="number"
          value={steps}
          onChange={(e) => setSteps(Number(e.target.value))}
          min={1}
          max={365}
          disabled={loading}
          style={{ width: 70 }}
        />
        <span className="portfolio-label">days</span>
        <button onClick={handleRun} disabled={loading}>
          {loading ? "Running..." : "Compare"}
        </button>
      </div>

      {error && <div className="error-banner">{error}</div>}

      {results && (
        <div className="portfolio-results">
          <p className="portfolio-model-label">Model: <strong>{results.model_name}</strong></p>
          <table>
            <thead>
              <tr>
                <th>Ticker</th>
                <th>Current Price</th>
                <th>Predicted Price</th>
                <th>Predicted Change</th>
                <th>RMSE</th>
                <th>R2</th>
              </tr>
            </thead>
            <tbody>
              {results.results.map((r) => (
                <tr key={r.ticker}>
                  <td><strong>{r.ticker}</strong></td>
                  <td>${r.current_price.toFixed(2)}</td>
                  <td>${r.predicted_price.toFixed(2)}</td>
                  <td className={r.predicted_change >= 0 ? "positive" : "negative"}>
                    {r.predicted_change >= 0 ? "+" : ""}{r.predicted_change}%
                  </td>
                  <td>{r.rmse}</td>
                  <td>{r.r2}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
