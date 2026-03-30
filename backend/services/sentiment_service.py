import logging
import re
from typing import List, Dict
from datetime import datetime

logger = logging.getLogger(__name__)

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False

try:
    import yfinance as yf
    YF_NEWS_AVAILABLE = True
except ImportError:
    YF_NEWS_AVAILABLE = False


def _clean_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def get_news_sentiment(ticker: str, max_items: int = 10) -> Dict:
    """Fetch news for a ticker and analyze sentiment using VADER."""
    if not VADER_AVAILABLE:
        return {"error": "vaderSentiment not installed", "articles": [], "avg_sentiment": 0}

    if not YF_NEWS_AVAILABLE:
        return {"error": "yfinance not installed", "articles": [], "avg_sentiment": 0}

    analyzer = SentimentIntensityAnalyzer()
    articles = []

    try:
        stock = yf.Ticker(ticker)
        news = stock.news or []

        for item in news[:max_items]:
            title = item.get("title", "")
            publisher = item.get("publisher", "")
            link = item.get("link", "")
            pub_date = ""
            if "providerPublishTime" in item:
                pub_date = datetime.fromtimestamp(item["providerPublishTime"]).strftime("%Y-%m-%d %H:%M")

            text = _clean_text(title)
            if not text:
                continue

            scores = analyzer.polarity_scores(text)
            compound = scores["compound"]

            if compound >= 0.05:
                label = "positive"
            elif compound <= -0.05:
                label = "negative"
            else:
                label = "neutral"

            articles.append({
                "title": title,
                "publisher": publisher,
                "url": link,
                "date": pub_date,
                "sentiment_score": round(compound, 3),
                "sentiment_label": label,
            })
    except Exception as e:
        logger.error("Failed to fetch news for %s: %s", ticker, e)
        return {"error": str(e), "articles": [], "avg_sentiment": 0}

    avg = sum(a["sentiment_score"] for a in articles) / len(articles) if articles else 0

    return {
        "ticker": ticker.upper(),
        "articles": articles,
        "avg_sentiment": round(avg, 3),
        "total_articles": len(articles),
    }
