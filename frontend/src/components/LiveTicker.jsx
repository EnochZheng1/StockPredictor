import { useState, useEffect, useRef } from "react";

export default function LiveTicker({ ticker }) {
  const [data, setData] = useState(null);
  const wsRef = useRef(null);

  useEffect(() => {
    if (!ticker) return;

    const wsUrl = (import.meta.env.VITE_API_URL || "http://localhost:4289/api")
      .replace(/^http/, "ws")
      .replace(/\/api$/, "");

    const ws = new WebSocket(`${wsUrl}/ws/price/${ticker}`);
    wsRef.current = ws;

    ws.onmessage = (event) => {
      try {
        const parsed = JSON.parse(event.data);
        if (!parsed.error) setData(parsed);
      } catch {}
    };

    ws.onerror = () => {};

    return () => {
      ws.close();
      wsRef.current = null;
    };
  }, [ticker]);

  if (!data || !data.price) return null;

  const isUp = (data.change || 0) >= 0;

  return (
    <div className={`live-ticker ${isUp ? "live-up" : "live-down"}`}>
      <span className="live-ticker-symbol">{data.ticker}</span>
      <span className="live-ticker-price">${data.price}</span>
      {data.change != null && (
        <span className="live-ticker-change">
          {isUp ? "+" : ""}{data.change} ({isUp ? "+" : ""}{data.change_pct}%)
        </span>
      )}
      <span className="live-ticker-dot" />
    </div>
  );
}
