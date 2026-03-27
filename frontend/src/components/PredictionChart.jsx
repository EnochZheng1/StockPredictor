import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ReferenceLine,
  ResponsiveContainer,
} from "recharts";

const COLORS = [
  "#2563eb",
  "#dc2626",
  "#16a34a",
  "#9333ea",
  "#ea580c",
  "#0891b2",
  "#be185d",
];

export default function PredictionChart({ stockData, comparisonResults }) {
  if (!stockData && !comparisonResults) return null;

  // Build chart data
  const chartData = [];
  let dividerDate = null;

  if (comparisonResults && comparisonResults.results.length > 0) {
    const first = comparisonResults.results[0];

    // Historical test period
    first.test_dates.forEach((date, i) => {
      const point = { date };
      if (stockData) {
        const idx = stockData.dates.indexOf(date);
        if (idx >= 0) point.actual = stockData.close_prices[idx];
      }
      comparisonResults.results.forEach((r) => {
        point[r.model_name] = r.test_predictions[i];
      });
      chartData.push(point);
    });

    dividerDate = first.test_dates[first.test_dates.length - 1];

    // Future predictions
    first.future_dates.forEach((date, i) => {
      const point = { date };
      comparisonResults.results.forEach((r) => {
        point[r.model_name] = r.future_predictions[i];
      });
      chartData.push(point);
    });
  } else if (stockData) {
    // Just show stock data
    const recent = Math.max(0, stockData.dates.length - 200);
    stockData.dates.slice(recent).forEach((date, i) => {
      chartData.push({
        date,
        actual: stockData.close_prices[recent + i],
      });
    });
  }

  // Thin out x-axis labels
  const tickInterval = Math.max(1, Math.floor(chartData.length / 10));

  const modelNames = comparisonResults
    ? comparisonResults.results.map((r) => r.model_name)
    : [];

  return (
    <div className="prediction-chart">
      <h3>Price Predictions</h3>
      <ResponsiveContainer width="100%" height={450}>
        <LineChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis
            dataKey="date"
            interval={tickInterval}
            tick={{ fontSize: 11 }}
            angle={-30}
            textAnchor="end"
            height={60}
          />
          <YAxis domain={["auto", "auto"]} tick={{ fontSize: 11 }} />
          <Tooltip />
          <Legend />
          {dividerDate && (
            <ReferenceLine
              x={dividerDate}
              stroke="#666"
              strokeDasharray="5 5"
              label="Forecast Start"
            />
          )}
          {stockData && comparisonResults && (
            <Line
              type="monotone"
              dataKey="actual"
              stroke="#000"
              strokeWidth={2}
              dot={false}
              name="Actual"
            />
          )}
          {modelNames.map((name, i) => (
            <Line
              key={name}
              type="monotone"
              dataKey={name}
              stroke={COLORS[i % COLORS.length]}
              strokeWidth={1.5}
              dot={false}
              name={name}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
