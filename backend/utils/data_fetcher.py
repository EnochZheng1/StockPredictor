import time
import logging
import yfinance as yf

logger = logging.getLogger(__name__)

_cache = {}
CACHE_TTL = 600  # 10 minutes


def get_historical_data(ticker, period="5y"):
    cache_key = f"{ticker}:{period}"
    now = time.time()

    if cache_key in _cache:
        data, timestamp = _cache[cache_key]
        if now - timestamp < CACHE_TTL:
            logger.info("Cache hit for %s (age: %.0fs)", cache_key, now - timestamp)
            return data.copy()

    logger.info("Fetching %s from yfinance (period=%s)", ticker, period)
    stock_data = yf.download(ticker, period=period)
    stock_data['Date'] = stock_data.index
    stock_data['SMA_20'] = stock_data['Adj Close'].rolling(window=20).mean()

    _cache[cache_key] = (stock_data, now)
    return stock_data


def clear_cache():
    _cache.clear()
    logger.info("Data cache cleared")
