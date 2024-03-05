import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

class linear_regression:
    def __init__(self, test_size=0.2, random_state=42):
        """
        Initialize the Linear Regression model and set parameters for data splitting.
        
        Parameters:
        - test_size: The proportion of the dataset to include in the test split.
        - random_state: Controls the shuffling applied to the data before applying the split.
        """
        self.model = LinearRegression()
        self.test_size = test_size
        self.random_state = random_state
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
    
    def split_data(self, X, y):
        """
        Split the data into training and testing sets.
        
        Parameters:
        - X: Features from the entire dataset.
        - y: Target variable from the entire dataset.
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)
        print("Data split into training and testing sets.")
    
    def fit(self):
        """
        Fit the Linear Regression model to the training data.
        Assumes that split_data has already been called to split the data into training and testing sets.
        """
        self.model.fit(self.X_train, self.y_train)
        print("Model training complete.")
    
    def predict(self, X=None):
        """
        Make predictions using the trained Linear Regression model.
        
        Parameters:
        - X: (Optional) Features from new data for which to make predictions. If None, predictions are made on the test set.
        
        Returns:
        - Predictions for the input data.
        """
        if X is None:
            X = self.X_test
        return self.model.predict(X)
    
    def evaluate(self):
        """
        Evaluate the performance of the Linear Regression model using the test set.
        Prints the Mean Squared Error and R-squared values.
        """
        predictions = self.predict()
        mse = mean_squared_error(self.y_test, predictions)
        r2 = r2_score(self.y_test, predictions)
        print(f"Mean Squared Error: {mse}")
        print(f"R-squared: {r2}")

    def predict_future(self, last_known_data, steps=10):
        """
        Predict future values based on the last known data and for a specified number of steps ahead.
        
        Parameters:
        - last_known_data: A numpy array or a pandas DataFrame with the last known data points.
        - steps: Number of future steps to predict.
        
        Returns:
        - A list of predicted values for the future steps.
        """
        future_predictions = []
        current_step_data = last_known_data.copy()
        
        for _ in range(steps):
            # Assuming the time feature is the last column in your dataset
            # Increment the time feature for the next prediction
            current_step_data[:, -1] += 1  # This line assumes your time feature is numeric and can be simply incremented
            
            # Predict the next step and append to future_predictions
            next_step_prediction = self.model.predict(current_step_data.reshape(1, -1))
            future_predictions.append(next_step_prediction.item())
            
            # Update current_step_data with the predicted value (if necessary)
            # This line is optional and depends on whether your future predictions rely on previous predictions
            
        return future_predictions