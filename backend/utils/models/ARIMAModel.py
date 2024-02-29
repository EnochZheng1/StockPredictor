import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt

class ARIMAModel:
    def __init__(self, order):
        """
        Initialize the ARIMA model with a specific order.
        
        Parameters:
        - order: A tuple (p, d, q) where p is the order (number of time lags) 
                 of the autoregressive model, d is the degree of differencing 
                 (the number of times the data have had past values subtracted), 
                 and q is the order of the moving-average model.
        """
        self.order = order
        self.model = None
        self.model_fit = None
    
    def fit(self, series):
        """
        Fit the ARIMA model to the provided time series.
        
        Parameters:
        - series: The time series data to fit the model to.
        """
        self.model = ARIMA(series, order=self.order)
        self.model_fit = self.model.fit()
        print("Model fitting complete.")
    
    def predict(self, start, end):
        """
        Make predictions with the fitted ARIMA model.
        
        Parameters:
        - start: The starting index for the predictions.
        - end: The ending index for the predictions.
        
        Returns:
        - Predictions from the model.
        """
        predictions = self.model_fit.predict(start=start, end=end)
        return predictions
    
    def evaluate(self, series, start, end):
        """
        Evaluate the ARIMA model's predictions against the actual values.
        
        Parameters:
        - series: The actual time series data for comparison.
        - start: The starting index for the predictions.
        - end: The ending index for the predictions.
        
        Prints the Root Mean Squared Error (RMSE) of the predictions.
        """
        predictions = self.predict(start, end)
        actual = series[start:end+1]  # Adjusted to include the 'end' index
        mse = mean_squared_error(actual, predictions)
        rmse = sqrt(mse)
        print(f"Root Mean Squared Error: {rmse}")
