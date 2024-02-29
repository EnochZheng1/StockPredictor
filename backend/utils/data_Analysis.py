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
    
def bollinger_bands(df, timeInterval=20, num_of_std=2):
    df['MA_{}'.format(timeInterval)] = df['Adj Close'].rolling(window=timeInterval).mean()
    df['STD_{}'.format(timeInterval)] = df['Adj Close'].rolling(window=timeInterval).std() 
    df['Upper_{}'.format(timeInterval)] = df['MA_{}'.format(timeInterval)] + (df['STD_{}'.format(timeInterval)] * num_of_std)
    df['Lower_{}'.format(timeInterval)] = df['MA_{}'.format(timeInterval)] - (df['STD_{}'.format(timeInterval)] * num_of_std)
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
    peak_price = df['High'].max()
    trough_price = df['Low'].min()
    price_range = peak_price - trough_price
    for level in fib_levels:
        df[f'Fib_{int(level*100)}%'] = trough_price + (price_range * level)
    return df

def on_balance_volume(df, diff=1):
    df['Volume_diff'] = df['Volume'].diff(diff)
    df['OBV'] = (np.where(df['Adj Close'] > df['Adj Close'].shift(diff), df['Volume'], 
              np.where(df['Adj Close'] < df['Adj Close'].shift(diff), -df['Volume'], 0))).cumsum()
    return df

def momentum(df, timeInterval=14):
    df['Momentum'] = df['Adj Close'].diff(timeInterval)
    return df

def average_true_range(df, timeInterval=14):
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Adj Close'].shift())
    low_close = np.abs(df['Low'] - df['Adj Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['ATR'] = true_range.rolling(window=timeInterval).mean()
    return df

def ichimoku_cloud(df):
    high_9 = df['High'].rolling(window=9).max()
    low_9 = df['Low'].rolling(window=9).min()
    df['Tenkan_sen'] = (high_9 + low_9) / 2

    high_26 = df['High'].rolling(window=26).max()
    low_26 = df['Low'].rolling(window=26).min()
    df['Kijun_sen'] = (high_26 + low_26) / 2

    df['Senkou_span_A'] = ((df['Tenkan_sen'] + df['Kijun_sen']) / 2).shift(26)

    high_52 = df['High'].rolling(window=52).max()
    low_52 = df['Low'].rolling(window=52).min()
    df['Senkou_span_B'] = ((high_52 + low_52) / 2).shift(26)

    df['Chikou_span'] = df['Adj Close'].shift(-26)
    return df

def williams_r(df, timeInterval=14):
    highest_high = df['High'].rolling(window=timeInterval).max()
    lowest_low = df['Low'].rolling(window=timeInterval).min()
    df['Williams_%R'] = ((highest_high - df['Adj Close']) / (highest_high - lowest_low)) * -100
    return df
    
def technical_indictors_calculation(df):
    # Calculate the 20-day Simple Moving Average (SMA)
    simple_moving_average(df)
    
    # Calculate the 50-day Simple Moving Average (SMA)
    simple_moving_average(df, 50)
    
    # Calculate the 20-day Exponential Moving Average (EMA)
    exponential_moving_average(df)
    
    # Calculate the 50-day Exponential Moving Average (EMA)
    exponential_moving_average(df, 50)
    
    # Calculate the Relative Strength Index (RSI)
    rsi(df)
    
    # Short term Exponential Moving Average
    # Long term Exponential Moving Average
    # Calculate the Moving Average Convergence Divergence (MACD)
    moving_average_convergence_divergence(df)
    
    # Calculate the Std Deviation, Upper Band and Lower Band
    bollinger_bands(df)
    
    # Calculate the Parabolic SAR (Stop and Reverse)
    parabolic_SAR(df)
    
    # Calculate the Average Directional Movement Index (ADX)
    average_directional_movement_index(df)
    
    # Calculate the Stochastic Oscillator
    stochastic_oscillator(df)
    
    # Calculate the price for each Fibonacci Level
    # fibonacci_level(df)
    
    return(df)