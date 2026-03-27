import { useState } from "react";

export default function TickerInput({ onFetch, loading }) {
  const [ticker, setTicker] = useState("AAPL");

  const handleSubmit = (e) => {
    e.preventDefault();
    if (ticker.trim()) onFetch(ticker.trim().toUpperCase());
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
      <button type="submit" disabled={loading || !ticker.trim()}>
        {loading ? "Loading..." : "Fetch Data"}
      </button>
    </form>
  );
}
