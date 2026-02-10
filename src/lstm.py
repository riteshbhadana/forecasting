import torch
import torch.nn as nn
import numpy as np


# ------------------------
# LSTM Model
# ------------------------
class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=32, batch_first=True)
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


# ------------------------
# Create sequences
# ------------------------
def create_sequences(data, seq_len=10):
    xs, ys = [], []
    for i in range(len(data) - seq_len):
        xs.append(data[i:i+seq_len])
        ys.append(data[i+seq_len])
    return np.array(xs), np.array(ys)


# ------------------------
# Train LSTM
# ------------------------
def train_lstm(series):
    data = series.values.astype(float)

    xs, ys = create_sequences(data)

    xs = torch.tensor(xs).float().unsqueeze(-1)
    ys = torch.tensor(ys).float().unsqueeze(-1)

    model = LSTMModel()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    for epoch in range(20):
        pred = model(xs)
        loss = loss_fn(pred, ys)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), "models/lstm.pt")

    return model
