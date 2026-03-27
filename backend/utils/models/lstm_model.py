import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from utils.models.base_model import BaseModel


class LSTMModel(BaseModel, nn.Module):

    def __init__(self, hidden_layer_size=64, sequence_length=10, num_layers=1, epochs=25, lr=0.01):
        BaseModel.__init__(self)
        nn.Module.__init__(self)
        self.hidden_layer_size = hidden_layer_size
        self.sequence_length = sequence_length
        self.num_layers = num_layers
        self.epochs = epochs
        self.lr = lr
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.target_scaler = MinMaxScaler(feature_range=(0, 1))
        self._input_size = None
        self._built = False

    def _build(self, input_size):
        self._input_size = input_size
        self.lstm = nn.LSTM(input_size, self.hidden_layer_size, self.num_layers, batch_first=True)
        self.linear = nn.Linear(self.hidden_layer_size, 1)
        self._built = True

    def get_name(self) -> str:
        return "LSTM"

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_layer_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_layer_size)
        out, _ = self.lstm(x, (h0, c0))
        out = self.linear(out[:, -1, :])
        return out

    def _create_sequences(self, X_scaled, y_scaled):
        sequences, labels = [], []
        for i in range(len(X_scaled) - self.sequence_length):
            sequences.append(X_scaled[i:i + self.sequence_length])
            labels.append(y_scaled[i + self.sequence_length])
        return np.array(sequences), np.array(labels)

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        X_values = X.values.astype(np.float32)
        y_values = y.values.reshape(-1, 1).astype(np.float32)

        X_scaled = self.scaler.fit_transform(X_values)
        y_scaled = self.target_scaler.fit_transform(y_values).flatten()

        if not self._built:
            self._build(X_values.shape[1])

        X_seq, y_seq = self._create_sequences(X_scaled, y_scaled)
        X_tensor = torch.tensor(X_seq, dtype=torch.float32)
        y_tensor = torch.tensor(y_seq, dtype=torch.float32).unsqueeze(1)

        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        nn.Module.train(self)
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            output = self.forward(X_tensor)
            loss = loss_fn(output, y_tensor)
            loss.backward()
            optimizer.step()

        self._last_X_scaled = X_scaled

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self.eval()
        X_values = X.values.astype(np.float32)
        X_scaled = self.scaler.transform(X_values)

        # Combine last sequence_length from training with test data
        full = np.vstack([self._last_X_scaled[-self.sequence_length:], X_scaled])

        predictions = []
        with torch.no_grad():
            for i in range(len(X_scaled)):
                seq = full[i:i + self.sequence_length]
                seq_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)
                pred = self.forward(seq_tensor)
                predictions.append(pred.item())

        predictions = np.array(predictions).reshape(-1, 1)
        return self.target_scaler.inverse_transform(predictions).flatten()

    def predict_future(self, last_known_data: pd.DataFrame, steps: int = 30) -> list:
        self.eval()
        X_values = last_known_data.values.astype(np.float32)
        current_seq = self.scaler.transform(X_values)[-self.sequence_length:]

        predictions = []
        with torch.no_grad():
            for _ in range(steps):
                seq_tensor = torch.tensor(current_seq, dtype=torch.float32).unsqueeze(0)
                pred_scaled = self.forward(seq_tensor).item()

                pred_price = self.target_scaler.inverse_transform([[pred_scaled]])[0][0]
                predictions.append(float(pred_price))

                # Roll the sequence forward with the new prediction
                new_row = current_seq[-1].copy()
                new_row[0] = pred_scaled  # Put predicted value in first feature slot
                current_seq = np.vstack([current_seq[1:], new_row])

        return predictions
