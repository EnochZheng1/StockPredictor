#import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import os
from utils.data_analysis import *
from utils.data_fetcher import *

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
#plot_data(analysis)