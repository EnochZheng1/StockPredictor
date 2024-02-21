from flask import Flask, request, jsonify
from flask_cors import CORS
import yfinance as yf
from sklearn.linear_model import LinearRegression
import numpy as np

# app = Flask(__name__)
# CORS(app)

def get_historical_data(ticker):
    stock_data = yf.download(ticker)
    stock_data['Date'] = stock_data.index
    return stock_data

def predict_future_prices(data):
    data['Days'] = (data['Date'] - data['Date'].min()).dt.days
    X = data[['Days']]
    y = data['Close']
    model = LinearRegression()
    model.fit(X, y)
    future_days = np.array([x for x in range(data['Days'].max()+1, data['Days'].max()+31)]).reshape(-1, 1)
    predictions = model.predict(future_days)
    return predictions

predicts = predict_future_prices(get_historical_data("AAPL"))
print(predicts)

# @app.route('/predict', methods=['POST'])
# def predict():
#     data = request.json
#     ticker = data['ticker']
#     stock_data = get_historical_data(ticker)
#     predictions = predict_future_prices(stock_data)
#     response = {
#         'historical': stock_data['Close'].tolist(),
#         'predictions': predictions.tolist()
#     }
#     return jsonify(response)

# if __name__ == '__main__':
#     app.run(debug=True)
