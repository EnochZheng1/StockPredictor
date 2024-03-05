import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class LongShortTermMemory:
    def __init__(self, input_size, hidden_layer_size, sequence_length, output_size=1, num_layers=1):
        self.input_size = input_size
        self.hidden_layer_size = hidden_layer_size
        self.sequence_length = sequence_length
        self.output_size = output_size
        self.num_layers = num_layers
        self.scaler = MinMaxScaler(feature_range=(0, 1))

        # Define the LSTM model
        self.model = nn.LSTM(input_size, hidden_layer_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.hidden = (torch.zeros(num_layers, 1, hidden_layer_size),
                       torch.zeros(num_layers, 1, hidden_layer_size))

    def forward(self, sequence):
        lstm_out, self.hidden = self.model(sequence.view(len(sequence), 1, -1), self.hidden)
        prediction = self.linear(lstm_out.view(len(sequence), -1))
        return prediction[-1]

    def train_model(self, train_data, epochs=100, lr=0.01):
        # Preprocess and scale the training data
        scaled_data = self.scaler.fit_transform(train_data)
        X, y = self.create_sequences(scaled_data)
        X_train = torch.tensor(X, dtype=torch.float32)
        y_train = torch.tensor(y, dtype=torch.float32)

        # Loss function and optimizer
        loss_function = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        # Training loop
        for i in range(epochs):
            for seq, labels in zip(X_train, y_train):
                optimizer.zero_grad()
                self.model.hidden = (torch.zeros(self.num_layers, 1, self.hidden_layer_size),
                                     torch.zeros(self.num_layers, 1, self.hidden_layer_size))

                y_pred = self.forward(seq)

                single_loss = loss_function(y_pred, labels)
                single_loss.backward()
                optimizer.step()

            if i % 10 == 0:
                print(f'Epoch {i} loss: {single_loss.item()}')

    def predict(self, recent_data):
        self.model.eval()
        with torch.no_grad():
            scaled_data = self.scaler.transform(recent_data)
            input_seq = scaled_data[-self.sequence_length:]
            input_seq_tensor = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0)

            self.model.hidden = (torch.zeros(self.num_layers, 1, self.hidden_layer_size),
                                 torch.zeros(self.num_layers, 1, self.hidden_layer_size))
            prediction = self.forward(input_seq_tensor)
            predicted_price = self.scaler.inverse_transform(prediction.numpy().reshape(-1, 1))

        return predicted_price

    def create_sequences(self, data):
        sequences = []
        labels = []
        for i in range(len(data) - self.sequence_length):
            seq = data[i:i + self.sequence_length]
            label = data[i + self.sequence_length][0]
            sequences.append(seq)
            labels.append(label)
        return np.array(sequences), np.array(labels)

    def predict_next_n_days(self, recent_data, n=10):
        predictions = []
        current_input = recent_data[-self.sequence_length:].copy()
        
        for _ in range(n):
            # Scale the current input
            scaled_current_input = self.scaler.transform(current_input)
            input_seq_tensor = torch.tensor(scaled_current_input, dtype=torch.float32).unsqueeze(0)
            
            # Predict the next day and scale back the prediction
            self.model.eval()
            with torch.no_grad():
                self.model.hidden = (torch.zeros(self.num_layers, 1, self.hidden_layer_size),
                                     torch.zeros(self.num_layers, 1, self.hidden_layer_size))
                prediction = self.forward(input_seq_tensor)
                predicted_price = self.scaler.inverse_transform(prediction.numpy().reshape(-1, 1))
                
            predictions.append(predicted_price.item())
            
            # Append the prediction to current_input and remove the first element
            # to maintain the sequence length
            next_day_input = np.array([predicted_price.item()])  # Assume single feature for simplicity
            current_input = np.vstack((current_input[1:], next_day_input))
        
        return predictions

# Example usage:
# Assuming your data is loaded into a DataFrame `df` with columns ['Price', 'Indicator1', ...]
# input_size = number of features (e.g., if you have 'Price' and 'Indicator1', input_size=2)
# sequence_length = number of time steps to look back

# predictor = LongShortTermMemory(input_size=2, hidden_layer_size=128, sequence_length=10, num_layers=2)
# predictor.train_model(train_data=df[['Price', 'Indicator1']].values, epochs=150, lr=0.001)
# future_price = predictor.predict(recent_data=df[['Price', 'Indicator1']].values[-10:])
# print(f"Predicted future price: {future_price}")

# Assuming your recent_data is loaded and formatted correctly, e.g., as a NumPy array
# recent_data = df[['Price', 'Indicator1']].values  # Just an example

# Assuming you've already instantiated and trained a StockPredictor object named `predictor`
# next_10_days_predictions = predictor.predict_next_n_days(recent_data, n=10)

# print("Predictions for the next 10 days:", next_10_days_predictions)