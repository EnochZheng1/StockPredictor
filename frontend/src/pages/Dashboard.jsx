import { useState, useEffect } from "react";
import TickerInput from "../components/TickerInput";
import ModelSelector from "../components/ModelSelector";
import PredictionChart from "../components/PredictionChart";
import ComparisonTable from "../components/ComparisonTable";
import LoadingSpinner from "../components/LoadingSpinner";
import {
  fetchStockData,
  getAvailableModels,
  runComparison,
} from "../api/stockApi";

export default function Dashboard() {
  const [models, setModels] = useState([]);
  const [ensembleMethods, setEnsembleMethods] = useState([]);
  const [stockData, setStockData] = useState(null);
  const [comparisonResults, setComparisonResults] = useState(null);
  const [ticker, setTicker] = useState("");
  const [fetchingStock, setFetchingStock] = useState(false);
  const [runningModels, setRunningModels] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    getAvailableModels()
      .then(({ models, ensembleMethods }) => {
        setModels(models);
        setEnsembleMethods(ensembleMethods);
      })
      .catch(() => setError("Failed to load available models"));
  }, []);

  const handleFetchStock = async (t) => {
    setFetchingStock(true);
    setError(null);
    setComparisonResults(null);
    try {
      const data = await fetchStockData(t);
      setStockData(data);
      setTicker(t);
    } catch (err) {
      setError(err.response?.data?.detail || "Failed to fetch stock data");
    } finally {
      setFetchingStock(false);
    }
  };

  const handleRunComparison = async (selectedModels, steps, selectedEnsembles = []) => {
    if (!ticker) {
      setError("Please fetch stock data first");
      return;
    }
    setRunningModels(true);
    setError(null);
    try {
      const results = await runComparison(ticker, selectedModels, steps, selectedEnsembles);
      setComparisonResults(results);
    } catch (err) {
      setError(err.response?.data?.detail || "Failed to run models");
    } finally {
      setRunningModels(false);
    }
  };

  return (
    <div className="dashboard">
      <div className="controls">
        <TickerInput onFetch={handleFetchStock} loading={fetchingStock} />
        {ticker && (
          <p className="ticker-label">
            Showing data for: <strong>{ticker}</strong>
          </p>
        )}
        {models.length > 0 && (
          <ModelSelector
            models={models}
            ensembleMethods={ensembleMethods}
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
      {(fetchingStock || runningModels) && (
        <LoadingSpinner
          message={
            fetchingStock
              ? "Fetching stock data..."
              : "Running models (this may take a minute)..."
          }
        />
      )}

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
    </div>
  );
}
