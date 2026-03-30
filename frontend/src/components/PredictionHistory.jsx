import { useState, useEffect } from "react";
import { fetchHistory } from "../api/stockApi";

export default function PredictionHistory({ ticker }) {
  const [history, setHistory] = useState([]);
  const [expanded, setExpanded] = useState(false);

  useEffect(() => {
    if (ticker) {
      fetchHistory(ticker, 20).then(setHistory).catch(() => {});
    }
  }, [ticker]);

  if (history.length === 0) return null;

  return (
    <div className="prediction-history">
      <button
        className="history-toggle"
        onClick={() => setExpanded(!expanded)}
      >
        {expanded ? "Hide" : "Show"} Prediction History ({history.length})
      </button>
      {expanded && (
        <table>
          <thead>
            <tr>
              <th>Date</th>
              <th>Model</th>
              <th>Period</th>
              <th>RMSE</th>
              <th>MAE</th>
              <th>R2</th>
            </tr>
          </thead>
          <tbody>
            {history.map((row) => (
              <tr key={row.id}>
                <td>{row.created_at?.slice(0, 16).replace("T", " ")}</td>
                <td>{row.model_name}</td>
                <td>{row.period}</td>
                <td>{row.rmse?.toFixed(4)}</td>
                <td>{row.mae?.toFixed(4)}</td>
                <td>{row.r2?.toFixed(4)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
