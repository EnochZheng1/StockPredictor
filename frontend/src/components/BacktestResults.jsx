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

const COLORS = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2", "#be185d"];

export default function BacktestResults({ backtestData }) {
  if (!backtestData?.results?.length) return null;

  // Build equity curve chart data
  const first = backtestData.results[0];
  const chartData = first.dates.map((date, i) => {
    const point = { date };
    backtestData.results.forEach((r) => {
      point[r.model_name] = r.equity_curve[i];
    });
    return point;
  });

  const tickInterval = Math.max(1, Math.floor(chartData.length / 10));

  return (
    <div className="backtest-results">
      <h3>Backtest Results (Long/Flat Strategy)</h3>
      <p className="backtest-subtitle">
        Buy & Hold Return: <strong>{backtestData.buy_hold_return}%</strong>
        {" | "}Best Strategy: <strong>{backtestData.best_strategy}</strong>
      </p>

      <div className="backtest-chart">
        <h4>Equity Curve (starting at $1)</h4>
        <ResponsiveContainer width="100%" height={350}>
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
            <YAxis tick={{ fontSize: 11 }} />
            <Tooltip />
            <Legend />
            <ReferenceLine y={1} stroke="#666" strokeDasharray="3 3" label="Start" />
            {backtestData.results.map((r, i) => (
              <Line
                key={r.model_name}
                type="monotone"
                dataKey={r.model_name}
                stroke={COLORS[i % COLORS.length]}
                strokeWidth={1.5}
                dot={false}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>

      <table className="backtest-table">
        <thead>
          <tr>
            <th>Model</th>
            <th>Return</th>
            <th>vs B&H</th>
            <th>Sharpe</th>
            <th>Max DD</th>
            <th>Win Rate</th>
            <th>Trades</th>
          </tr>
        </thead>
        <tbody>
          {backtestData.results.map((r) => {
            const alpha = r.total_return - backtestData.buy_hold_return;
            return (
              <tr key={r.model_name} className={r.model_name === backtestData.best_strategy ? "best-row" : ""}>
                <td>
                  {r.model_name}
                  {r.model_name === backtestData.best_strategy && (
                    <span className="best-badge">Best</span>
                  )}
                </td>
                <td className={r.total_return >= 0 ? "positive" : "negative"}>
                  {r.total_return}%
                </td>
                <td className={alpha >= 0 ? "positive" : "negative"}>
                  {alpha >= 0 ? "+" : ""}{alpha.toFixed(2)}%
                </td>
                <td>{r.sharpe_ratio}</td>
                <td>{r.max_drawdown}%</td>
                <td>{r.win_rate}%</td>
                <td>{r.num_trades}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
