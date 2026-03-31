import { useState, useEffect } from "react";
import { fetchSentiment } from "../api/stockApi";

const LABEL_COLORS = {
  positive: "#16a34a",
  negative: "#dc2626",
  neutral: "#64748b",
};

export default function SentimentPanel({ ticker }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!ticker) return;
    setLoading(true);
    setError(null);
    fetchSentiment(ticker)
      .then(setData)
      .catch(() => setError("Failed to load sentiment data"))
      .finally(() => setLoading(false));
  }, [ticker]);

  if (!ticker || loading) return null;
  if (error) return null;
  if (!data || data.error || !data.articles?.length) return null;

  const sentimentColor = data.avg_sentiment >= 0.05
    ? "#16a34a"
    : data.avg_sentiment <= -0.05
      ? "#dc2626"
      : "#64748b";

  return (
    <div className="sentiment-panel">
      <div className="sentiment-header">
        <h3>News Sentiment</h3>
        <span className="sentiment-score" style={{ color: sentimentColor }}>
          Avg: {data.avg_sentiment > 0 ? "+" : ""}{data.avg_sentiment}
        </span>
      </div>
      <div className="sentiment-articles">
        {data.articles.map((article) => (
          <div key={article.url || article.title} className="sentiment-article">
            <div className="sentiment-article-header">
              <span
                className="sentiment-badge"
                style={{ background: LABEL_COLORS[article.sentiment_label] || "#64748b" }}
              >
                {article.sentiment_label} ({article.sentiment_score})
              </span>
              <span className="sentiment-date">{article.date}</span>
            </div>
            <a
              href={article.url}
              target="_blank"
              rel="noopener noreferrer"
              className="sentiment-title"
            >
              {article.title}
            </a>
            <span className="sentiment-publisher">{article.publisher}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
