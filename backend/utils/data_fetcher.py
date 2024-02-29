import yfinance as yf

def get_historical_data(ticker):
    stock_data = yf.download(ticker)
    stock_data['Date'] = stock_data.index
    stock_data['SMA_20'] = stock_data['Adj Close'].rolling(window=20).mean()
    return stock_data