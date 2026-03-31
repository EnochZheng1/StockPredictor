export default function ComparisonTable({ summary, bestModel }) {
  if (!summary || summary.length === 0) return null;

  return (
    <div className="comparison-table">
      <h3>Model Comparison</h3>
      <table>
        <thead>
          <tr>
            <th>Model</th>
            <th>RMSE</th>
            <th>MAE</th>
            <th>R2</th>
          </tr>
        </thead>
        <tbody>
          {summary.map((row) => (
            <tr
              key={row.model_name}
              className={`${row.model_name === bestModel ? "best-row" : ""} ${row.model_name.startsWith("Ensemble") ? "ensemble-row" : ""}`}
            >
              <td>
                {row.model_name}
                {row.model_name === bestModel && (
                  <span className="best-badge">Best</span>
                )}
              </td>
              <td>{Number(row.rmse ?? 0).toFixed(4)}</td>
              <td>{Number(row.mae ?? 0).toFixed(4)}</td>
              <td>{Number(row.r2 ?? 0).toFixed(4)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
