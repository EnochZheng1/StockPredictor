import numpy as np
import pandas as pd
import talib

def simple_moving_average(df, timeInterval=20):
    df["SMA_{}".format(timeInterval)] = df['Adj Close'].rolling(window=timeInterval).mean()
    return df

def exponential_moving_average(df, timeInterval=20):
    df["EMA_{}".format(timeInterval)] = df['Adj Close'].ewm(span=timeInterval, adjust=False).mean()
    return df

def rsi(df, timeInterval=14):
    delta = df['Adj Close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=timeInterval).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=timeInterval).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df
    
def moving_average_convergence_divergence(df, shortTermInterval=12, longTermInterval=26, signLineInterval=9):
    ShortEMA = df['Adj Close'].ewm(span=shortTermInterval, adjust=False).mean()
    LongEMA = df['Adj Close'].ewm(span=longTermInterval, adjust=False).mean()
    df['MACD'] = ShortEMA - LongEMA
    df['Signal_Line'] = df['MACD'].ewm(span=signLineInterval, adjust=False).mean()
    return df
    
def std_upper_lower_bound(df, timeInterval=20):
    df['{}STD'.format(timeInterval)] = df['Adj Close'].rolling(window=timeInterval).std()
    df['{}Upper'.format(timeInterval)] = df['Adj Close'].rolling(window=timeInterval).mean() + (df['{}STD'.format(timeInterval)] * 2)
    df['{}Lower'.format(timeInterval)] = df['Adj Close'].rolling(window=timeInterval).mean() - (df['{}STD'.format(timeInterval)] * 2)
    return df
    
def parabolic_SAR(df, acceleration=0.02, maximum=0.2):
    df['SAR'] = talib.SAR(df['High'], df['Low'], acceleration=acceleration, maximum=maximum)
    return df
    
def average_directional_movement_index(df, timeInterval=14):
    df['ADX'] = talib.ADX(df['High'], df['Low'], df['Close'], timeperiod=timeInterval)
    return df
    
def stochastic_oscillator(df, fastk_period=14, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0):
    df['SlowK'], df['SlowD'] = talib.STOCH(df['High'], df['Low'], df['Close'],
                           fastk_period=fastk_period, slowk_period=slowk_period, slowk_matype=slowk_matype,
                           slowd_period=slowd_period, slowd_matype=slowd_matype)
    return df
    
def fibonacci_level(df, fib_levels=[0, 0.236, 0.382, 0.618, 0.786, 1]):
    # Calculate the price for each Fibonacci Level
    peak_price = df['High'].max()
    trough_price = df['Low'].min()
    price_range = peak_price - trough_price
    for level in fib_levels:
        df[f'Fib_{int(level*100)}%'] = trough_price + (price_range * level)
    return df
    


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