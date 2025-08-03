
# 📈 Time Series Stock Forecasting Dashboard

This project is a complete end-to-end time series forecasting system that compares multiple predictive models for **Apple Inc. (AAPL)** stock prices. It includes data preprocessing, model training, evaluation, and a dynamic dashboard for visualizing forecasts and performance metrics.

---

## 🧠 Models Included

| Model    | Description |
|----------|-------------|
| **ARIMA**   | Classical autoregressive model for stationary data |
| **SARIMA**  | Seasonal ARIMA to handle time-based seasonality |
| **Prophet** | Facebook’s forecasting model for flexible trend/seasonality handling |
| **LSTM**    | Deep learning model using memory cells to learn long-term dependencies |

---

## 📊 Features

✅ Cleaned historical data from Yahoo Finance  
✅ Monthly-level aggregation for better signal extraction  
✅ Forecast generation using 4 time series models  
✅ Interactive Streamlit dashboard with:
- Line charts for historical and predicted values
- Side-by-side model comparison
- Evaluation metrics: MAE, RMSE, MAPE
- Upload your own forecast images  
✅ Modular architecture for adding more models

---

## 🗂️ Project Structure

```

time-series-stock-forecasting/
│
├── app.py                # 📲 Streamlit dashboard
├── data/                 # 📁 Cleaned input data
│   └── AAPL\_cleaned.csv
│
├── models/               # 🧠 All 4 forecasting models
│   ├── arima\_model.py
│   ├── sarima\_model.py
│   ├── prophet\_model.py
│   └── lstm\_model.py
│
├── forecasts/            # 📈 Optional: CSV forecasted outputs
├── metrics/              # 📊 Evaluation metrics in JSON
├── images/               # 🖼️ Forecast plots for each model
├── requirements.txt      # 📦 Python dependencies
└── README.md             # 📘 Project overview

````

---

## 🚀 Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/your-username/time-series-stock-forecasting.git
cd time-series-stock-forecasting
````

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\\Scripts\\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit app

```bash
streamlit run app.py
```

---

## 📉 Evaluation Metrics

Each model outputs:

* **MAE (Mean Absolute Error)**: Average magnitude of error
* **RMSE (Root Mean Squared Error)**: Penalizes larger errors
* **MAPE (Mean Absolute Percentage Error)**: Scaled error

These are shown in the dashboard below each forecast to help choose the most accurate model.

---

## 📌 Next Steps

🔹 Add more models (e.g., XGBoost, Transformer-based forecasting)
🔹 Integrate trade signal logic (Buy/Sell based on trend)
🔹 Allow custom ticker input
🔹 Deploy online via Streamlit Cloud or Heroku

---


## ✨ Acknowledgements

* [yfinance](https://github.com/ranaroussi/yfinance) for stock data
* [Prophet](https://facebook.github.io/prophet/) by Meta
* [Statsmodels](https://www.statsmodels.org/)
* [TensorFlow / Keras](https://www.tensorflow.org/)

```

