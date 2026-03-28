import { useState } from "react";

const PERIODS = [
  { value: "1y", label: "1 Year" },
  { value: "2y", label: "2 Years" },
  { value: "5y", label: "5 Years" },
  { value: "10y", label: "10 Years" },
  { value: "max", label: "Max" },
];

export default function TickerInput({ onFetch, loading }) {
  const [ticker, setTicker] = useState("AAPL");
  const [period, setPeriod] = useState("5y");

  const handleSubmit = (e) => {
    e.preventDefault();
    if (ticker.trim()) onFetch(ticker.trim().toUpperCase(), period);
  };

  return (
    <form onSubmit={handleSubmit} className="ticker-input">
      <input
        type="text"
        value={ticker}
        onChange={(e) => setTicker(e.target.value)}
        placeholder="Enter ticker (e.g. AAPL)"
        disabled={loading}
      />
      <select value={period} onChange={(e) => setPeriod(e.target.value)} disabled={loading}>
        {PERIODS.map((p) => (
          <option key={p.value} value={p.value}>{p.label}</option>
        ))}
      </select>
      <button type="submit" disabled={loading || !ticker.trim()}>
        {loading ? "Loading..." : "Fetch Data"}
      </button>
    </form>
  );
}
