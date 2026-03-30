import { useState, useEffect } from "react";
import TickerInput from "../components/TickerInput";
import ModelSelector from "../components/ModelSelector";
import PredictionChart from "../components/PredictionChart";
import ComparisonTable from "../components/ComparisonTable";
import FeatureImportanceChart from "../components/FeatureImportanceChart";
import ExportButton from "../components/ExportButton";
import BacktestResults from "../components/BacktestResults";
import CandlestickChart from "../components/CandlestickChart";
import PredictionHistory from "../components/PredictionHistory";
import PortfolioView from "../components/PortfolioView";
import SentimentPanel from "../components/SentimentPanel";
import LiveTicker from "../components/LiveTicker";
import LoadingSpinner from "../components/LoadingSpinner";
import {
  fetchStockData,
  getAvailableModels,
  runComparison,
  runBacktest,
} from "../api/stockApi";

export default function Dashboard() {
  const [models, setModels] = useState([]);
  const [ensembleMethods, setEnsembleMethods] = useState([]);
  const [modelParams, setModelParams] = useState({});
  const [stockData, setStockData] = useState(null);
  const [comparisonResults, setComparisonResults] = useState(null);
  const [backtestData, setBacktestData] = useState(null);
  const [ticker, setTicker] = useState("");
  const [period, setPeriod] = useState("5y");
  const [lastSelectedModels, setLastSelectedModels] = useState([]);
  const [lastParamOverrides, setLastParamOverrides] = useState({});
  const [fetchingStock, setFetchingStock] = useState(false);
  const [runningModels, setRunningModels] = useState(false);
  const [runningBacktest, setRunningBacktest] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    getAvailableModels()
      .then(({ models, ensembleMethods, modelParams }) => {
        setModels(models);
        setEnsembleMethods(ensembleMethods);
        setModelParams(modelParams);
      })
      .catch(() => setError("Failed to load available models"));
  }, []);

  const handleFetchStock = async (t, p) => {
    setFetchingStock(true);
    setError(null);
    setComparisonResults(null);
    setBacktestData(null);
    try {
      const data = await fetchStockData(t, p);
      setStockData(data);
      setTicker(t);
      setPeriod(p);
    } catch (err) {
      setError(err.response?.data?.detail || "Failed to fetch stock data");
    } finally {
      setFetchingStock(false);
    }
  };

  const handleRunComparison = async (selectedModels, steps, selectedEnsembles = [], paramOverrides = {}) => {
    if (!ticker) {
      setError("Please fetch stock data first");
      return;
    }
    setRunningModels(true);
    setError(null);
    setLastSelectedModels(selectedModels);
    setLastParamOverrides(paramOverrides);
    try {
      const results = await runComparison(ticker, selectedModels, steps, period, selectedEnsembles, paramOverrides);
      setComparisonResults(results);
      setBacktestData(null);
    } catch (err) {
      setError(err.response?.data?.detail || "Failed to run models");
    } finally {
      setRunningModels(false);
    }
  };

  const handleRunBacktest = async () => {
    if (!ticker || lastSelectedModels.length === 0) return;
    setRunningBacktest(true);
    setError(null);
    try {
      const data = await runBacktest(ticker, lastSelectedModels, period, lastParamOverrides);
      setBacktestData(data);
    } catch (err) {
      setError(err.response?.data?.detail || "Failed to run backtest");
    } finally {
      setRunningBacktest(false);
    }
  };

  return (
    <div className="dashboard">
      <div className="controls">
        <TickerInput onFetch={handleFetchStock} loading={fetchingStock} />
        {ticker && (
          <>
            <p className="ticker-label">
              Showing data for: <strong>{ticker}</strong>
            </p>
            <LiveTicker ticker={ticker} />
          </>
        )}
        {models.length > 0 && (
          <ModelSelector
            models={models}
            ensembleMethods={ensembleMethods}
            modelParams={modelParams}
            onRun={handleRunComparison}
            loading={runningModels}
          />
        )}
      </div>

      {error && <div className="error-banner">{error}</div>}
      {comparisonResults?.errors?.length > 0 && (
        <div className="warning-banner">
          Some models failed:
          <ul>
            {comparisonResults.errors.map((e, i) => <li key={i}>{e}</li>)}
          </ul>
        </div>
      )}
      {(fetchingStock || runningModels || runningBacktest) && (
        <LoadingSpinner
          message={
            fetchingStock
              ? "Fetching stock data..."
              : runningBacktest
                ? "Running backtest..."
                : "Running models (this may take a minute)..."
          }
        />
      )}

      <SentimentPanel ticker={ticker} />

      <CandlestickChart stockData={stockData} />

      <PredictionChart
        stockData={stockData}
        comparisonResults={comparisonResults}
      />

      {comparisonResults && (
        <ComparisonTable
          summary={comparisonResults.summary}
          bestModel={comparisonResults.best_model}
        />
      )}

      {comparisonResults && (
        <div className="backtest-actions">
          <ExportButton comparisonResults={comparisonResults} />
          <button
            className="backtest-btn"
            onClick={handleRunBacktest}
            disabled={runningBacktest}
          >
            {runningBacktest ? "Running Backtest..." : "Run Backtest"}
          </button>
        </div>
      )}

      <BacktestResults backtestData={backtestData} />

      <FeatureImportanceChart comparisonResults={comparisonResults} />

      <PredictionHistory ticker={ticker} />

      {models.length > 0 && <PortfolioView models={models} />}
    </div>
  );
}
