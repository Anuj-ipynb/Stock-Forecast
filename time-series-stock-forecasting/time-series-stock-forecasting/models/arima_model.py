import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import warnings
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import json
import os
warnings.filterwarnings("ignore")

df = pd.read_csv("AAPL_cleaned.csv", index_col=0)
df.index = pd.to_datetime(df.index)
monthly_df = df['Close'].resample('ME').mean().dropna()  # 'ME' replaces deprecated 'M'

model = ARIMA(monthly_df, order=(2,1,2))  # Simpler, safer
results = model.fit()

forecast = results.forecast(steps=12)

plt.figure(figsize=(12, 6))
plt.plot(monthly_df, label='Actual')
plt.plot(forecast.index, forecast, label='ARIMA Forecast', linestyle='--')
plt.title("ARIMA Forecast")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("arima_forecast.png")
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
