import talib
import yfinance as yf
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import os

def get_historical_data(ticker):
    stock_data = yf.download(ticker)
    stock_data['Date'] = stock_data.index
    stock_data['SMA_20'] = stock_data['Adj Close'].rolling(window=20).mean()
    return stock_data

def technical_indictors_calculation(df):
    # Calculate the 20-day Simple Moving Average (SMA)
    df['SMA_20'] = df['Adj Close'].rolling(window=20).mean()
    
    # Calculate the 50-day Simple Moving Average (SMA)
    df['SMA_50'] = df['Adj Close'].rolling(window=50).mean()
    
    # Calculate the 20-day Exponential Moving Average (EMA)
    df['EMA_20'] = df['Adj Close'].ewm(span=20, adjust=False).mean()
    
    # Calculate the 50-day Exponential Moving Average (EMA)
    df['EMA_50'] = df['Adj Close'].ewm(span=50, adjust=False).mean()
    
    # Calculate the Relative Strength Index (RSI)
    delta = df['Adj Close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Short term Exponential Moving Average
    ShortEMA = df['Adj Close'].ewm(span=12, adjust=False).mean()
    # Long term Exponential Moving Average
    LongEMA = df['Adj Close'].ewm(span=26, adjust=False).mean()
    # Calculate the Moving Average Convergence Divergence (MACD)
    df['MACD'] = ShortEMA - LongEMA
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Calculate the Std Deviation, Upper Band and Lower Band
    df['20STD'] = df['Adj Close'].rolling(window=20).std()
    df['Upper'] = df['SMA_20'] + (df['20STD'] * 2)
    df['Lower'] = df['SMA_20'] - (df['20STD'] * 2)
    
    # Calculate the Parabolic SAR (Stop and Reverse)
    df['SAR'] = talib.SAR(df['High'], df['Low'], acceleration=0.02, maximum=0.2)
    
    # Calculate the Average Directional Movement Index (ADX)
    df['ADX'] = talib.ADX(df['High'], df['Low'], df['Close'], timeperiod=14)
    
    # Calculate the Stochastic Oscillator
    df['SlowK'], df['SlowD'] = talib.STOCH(df['High'], df['Low'], df['Close'],
                           fastk_period=14, slowk_period=3, slowk_matype=0,
                           slowd_period=3, slowd_matype=0)
    
    # Calculate the price for each Fibonacci Level
    peak_price = df['High'].max()
    trough_price = df['Low'].min()
    price_range = peak_price - trough_price
    fib_levels = [0, 0.236, 0.382, 0.618, 0.786, 1]
    for level in fib_levels:
        df[f'Fib_{int(level*100)}%'] = trough_price + (price_range * level)
    
    return(df)

def plot_data(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Close', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA_20'], mode='lines', name='SMA_20', line=dict(color='red')))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['EMA_20'], mode='lines', name='EMA_20', line=dict(color='green')))
    fig.update_yaxes(type="log")
    fig.update_traces(mode='lines+markers')
    fig.update_layout(title='Close Price and Indicators Over Time',
                    xaxis_title='Date',
                    yaxis_title='Price',
                    legend_title='Indicators',
                    hovermode='x unified')
    fig.show()
# def simple_prediction_open_to_closing(df):
    

data = get_historical_data("AAPL")
analysis = technical_indictors_calculation(data)
if os.path.exists('data.csv'):
    os.remove('data.csv')
analysis.to_csv('data.csv', index=False)
print(analysis)
plot_data(analysis)