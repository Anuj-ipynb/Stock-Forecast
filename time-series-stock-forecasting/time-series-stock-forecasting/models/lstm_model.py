import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import warnings
from sklearn.metrics import mean_absolute_error, mean_squared_error

import json
import os
warnings.filterwarnings("ignore")

# Load your monthly stock price data
df = pd.read_csv("AAPL_cleaned.csv", index_col=0)
df.index = pd.to_datetime(df.index)
monthly = df['Close'].resample('ME').mean().dropna()  # 'ME' instead of deprecated 'M'

# Normalize
scaler = MinMaxScaler()
scaled = scaler.fit_transform(monthly.values.reshape(-1, 1))

# Create sequences
def create_dataset(data, time_step=12):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

time_step = 12
X, y = create_dataset(scaled, time_step)
X = X.reshape(X.shape[0], X.shape[1], 1)

# Build LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(time_step, 1)),
    LSTM(50),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X, y, epochs=10, batch_size=1, verbose=1)

# Forecast next 12 steps
x_input = scaled[-time_step:].reshape(1, time_step, 1)  # start with last 12 inputs
predictions = []

for _ in range(12):
    pred = model.predict(x_input, verbose=0)
    predictions.append(pred[0, 0])
    x_input = np.concatenate([x_input[:, 1:, :], pred.reshape(1, 1, 1)], axis=1)

# Inverse scale the forecast
forecast = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()

# Generate future dates
forecast_index = pd.date_range(start=monthly.index[-1] + pd.DateOffset(months=1), periods=12, freq='M')
forecast_series = pd.Series(forecast, index=forecast_index)

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(monthly, label="Historical")
plt.plot(forecast_series, label="LSTM Forecast", linestyle='--')
plt.title("LSTM Forecast")
plt.xlabel("Date")
plt.ylabel("Price")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("lstm_forecast.png")
plt.show()

# Ensure the 'metrics' directory exists
os.makedirs("metrics", exist_ok=True)

# Compute metrics (true and pred should be numpy arrays of equal length)
mae = mean_absolute_error(true, pred)
rmse = mean_squared_error(true, pred, squared=False)
mape = np.mean(np.abs((true - pred) / true)) * 100

metrics = {
    "MAE": float(mae),
    "RMSE": float(rmse),
    "MAPE": float(mape)
}

# Save to JSON
with open("metrics/XXX_metrics.json", "w") as f:  # Replace XXX with model name in lowercase
    json.dump(metrics, f)
