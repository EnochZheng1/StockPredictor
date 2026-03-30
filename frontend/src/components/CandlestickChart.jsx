import { useEffect, useRef, useState } from "react";
import { createChart } from "lightweight-charts";

export default function CandlestickChart({ stockData }) {
  const chartContainerRef = useRef(null);
  const chartRef = useRef(null);
  const [showVolume, setShowVolume] = useState(true);
  const [overlay, setOverlay] = useState("sma");

  useEffect(() => {
    if (!stockData || !chartContainerRef.current) return;

    // Clean up previous chart
    if (chartRef.current) {
      chartRef.current.remove();
      chartRef.current = null;
    }

    const isDark = document.documentElement.getAttribute("data-theme") === "dark";

    const chart = createChart(chartContainerRef.current, {
      width: chartContainerRef.current.clientWidth,
      height: 420,
      layout: {
        background: { color: isDark ? "#1e293b" : "#ffffff" },
        textColor: isDark ? "#94a3b8" : "#475569",
      },
      grid: {
        vertLines: { color: isDark ? "#334155" : "#e2e8f0" },
        horzLines: { color: isDark ? "#334155" : "#e2e8f0" },
      },
      timeScale: { timeVisible: false },
    });
    chartRef.current = chart;

    // Candlestick data
    const candles = stockData.dates.map((date, i) => ({
      time: date,
      open: stockData.open_prices[i],
      high: stockData.high_prices[i],
      low: stockData.low_prices[i],
      close: stockData.close_prices[i],
    })).filter((c) => c.open != null && c.close != null);

    const candleSeries = chart.addCandlestickSeries({
      upColor: "#16a34a",
      downColor: "#dc2626",
      borderUpColor: "#16a34a",
      borderDownColor: "#dc2626",
      wickUpColor: "#16a34a",
      wickDownColor: "#dc2626",
    });
    candleSeries.setData(candles);

    // Volume bars
    if (showVolume) {
      const volumeData = stockData.dates.map((date, i) => ({
        time: date,
        value: stockData.volume[i] || 0,
        color: stockData.close_prices[i] >= (stockData.open_prices[i] || 0)
          ? "rgba(22, 163, 74, 0.3)"
          : "rgba(220, 38, 38, 0.3)",
      })).filter((v) => v.value != null);

      const volumeSeries = chart.addHistogramSeries({
        priceFormat: { type: "volume" },
        priceScaleId: "volume",
      });
      volumeSeries.priceScale().applyOptions({
        scaleMargins: { top: 0.8, bottom: 0 },
      });
      volumeSeries.setData(volumeData);
    }

    // Overlay indicator
    const indicators = stockData.indicators || {};
    const overlayKey = overlay === "sma" ? "SMA_20" : overlay === "ema" ? "EMA_20" : null;
    if (overlayKey && indicators[overlayKey]) {
      const lineData = stockData.dates
        .map((date, i) => ({
          time: date,
          value: indicators[overlayKey][i],
        }))
        .filter((d) => d.value != null);

      const lineSeries = chart.addLineSeries({
        color: overlay === "sma" ? "#f59e0b" : "#8b5cf6",
        lineWidth: 1,
      });
      lineSeries.setData(lineData);
    }

    // Bollinger Bands
    if (overlay === "bollinger" && indicators["Upper_20"] && indicators["Lower_20"]) {
      const upperData = stockData.dates
        .map((date, i) => ({ time: date, value: indicators["Upper_20"][i] }))
        .filter((d) => d.value != null);
      const lowerData = stockData.dates
        .map((date, i) => ({ time: date, value: indicators["Lower_20"][i] }))
        .filter((d) => d.value != null);

      chart.addLineSeries({ color: "#3b82f6", lineWidth: 1, lineStyle: 2 }).setData(upperData);
      chart.addLineSeries({ color: "#3b82f6", lineWidth: 1, lineStyle: 2 }).setData(lowerData);
    }

    chart.timeScale().fitContent();

    const handleResize = () => {
      if (chartContainerRef.current) {
        chart.applyOptions({ width: chartContainerRef.current.clientWidth });
      }
    };
    window.addEventListener("resize", handleResize);
    return () => {
      window.removeEventListener("resize", handleResize);
      chart.remove();
      chartRef.current = null;
    };
  }, [stockData, showVolume, overlay]);

  if (!stockData) return null;

  return (
    <div className="candlestick-chart">
      <div className="candlestick-header">
        <h3>Price Chart</h3>
        <div className="candlestick-controls">
          <select value={overlay} onChange={(e) => setOverlay(e.target.value)}>
            <option value="none">No Overlay</option>
            <option value="sma">SMA 20</option>
            <option value="ema">EMA 20</option>
            <option value="bollinger">Bollinger Bands</option>
          </select>
          <label>
            <input
              type="checkbox"
              checked={showVolume}
              onChange={(e) => setShowVolume(e.target.checked)}
            />
            Volume
          </label>
        </div>
      </div>
      <div ref={chartContainerRef} />
    </div>
  );
}
