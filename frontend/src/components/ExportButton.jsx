export default function ExportButton({ comparisonResults }) {
  if (!comparisonResults || comparisonResults.results.length === 0) return null;

  const exportMetricsCSV = () => {
    const header = "Model,RMSE,MAE,R2";
    const rows = comparisonResults.summary.map(
      (r) => `${r.model_name},${r.rmse},${r.mae},${r.r2}`
    );
    download(`${header}\n${rows.join("\n")}`, "model_metrics.csv");
  };

  const exportPredictionsCSV = () => {
    const first = comparisonResults.results[0];
    const modelNames = comparisonResults.results.map((r) => r.model_name);
    const header = `Date,${modelNames.join(",")}`;

    const rows = [];
    // Test predictions
    first.test_dates.forEach((date, i) => {
      const values = comparisonResults.results.map((r) => r.test_predictions[i] ?? "");
      rows.push(`${date},${values.join(",")}`);
    });
    // Future predictions
    first.future_dates.forEach((date, i) => {
      const values = comparisonResults.results.map((r) => r.future_predictions[i] ?? "");
      rows.push(`${date},${values.join(",")}`);
    });

    download(`${header}\n${rows.join("\n")}`, "predictions.csv");
  };

  return (
    <div className="export-buttons">
      <button className="export-btn" onClick={exportMetricsCSV}>
        Export Metrics CSV
      </button>
      <button className="export-btn" onClick={exportPredictionsCSV}>
        Export Predictions CSV
      </button>
    </div>
  );
}

function download(content, filename) {
  const blob = new Blob([content], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}
