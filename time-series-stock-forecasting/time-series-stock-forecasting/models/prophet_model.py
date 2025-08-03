from prophet import Prophet
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import json
import os

df = pd.read_csv("AAPL_cleaned.csv", index_col=0)
df.index = pd.to_datetime(df.index)
monthly = df['Close'].resample('M').mean().dropna()

dfp = monthly.reset_index().rename(columns={"Date": "ds", "Close": "y"})

model = Prophet()
model.fit(dfp)

future = model.make_future_dataframe(periods=12, freq='M')
forecast = model.predict(future)

fig = model.plot(forecast)
plt.title("Prophet Forecast")
fig.savefig("prophet_forecast.png")


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
