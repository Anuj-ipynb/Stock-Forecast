import json

model_images = {
    "LSTM": "lstm_forecast.png",
    "ARIMA": "arima_forecast.png",
    "SARIMA": "sarima_forecast.png",
    "Prophet": "prophet_forecast.png"
}

metric_files = {
    "LSTM": "metrics/lstm_metrics.json",
    "ARIMA": "metrics/arima_metrics.json",
    "SARIMA": "metrics/sarima_metrics.json",
    "Prophet": "metrics/prophet_metrics.json"
}

tab_names = ["LSTM", "ARIMA", "SARIMA", "Prophet"]
tabs = st.tabs([f"📊 {name}" for name in tab_names])

for tab, model in zip(tabs, tab_names):
    with tab:
        st.markdown(f"### {model} Forecast")

        # 📈 Load and show image
        image_path = model_images.get(model)
        if os.path.exists(image_path):
            st.image(image_path, use_column_width=True)
        else:
            st.warning(f"{model} forecast plot not found. Please run `{model.lower()}_model.py` to generate it.")

        # 📊 Load and show metrics
        metric_path = metric_files.get(model)
        if os.path.exists(metric_path):
            with open(metric_path, "r") as f:
                metrics = json.load(f)
            st.markdown("#### 📉 Evaluation Metrics")
            col1, col2, col3 = st.columns(3)
            col1.metric("MAE", f"{metrics['MAE']:.2f}")
            col2.metric("RMSE", f"{metrics['RMSE']:.2f}")
            col3.metric("MAPE (%)", f"{metrics['MAPE']:.2f}")
        else:
            st.warning(f"{model} evaluation metrics not found. Please ensure `{model.lower()}_metrics.json` exists.")
