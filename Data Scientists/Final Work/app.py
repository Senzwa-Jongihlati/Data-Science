
import os
import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, "xgb_forecast_model.joblib"))
feature_columns = joblib.load(os.path.join(BASE_DIR, "feature_columns_fc.joblib"))

measurement_cols = [
    "pH", "Turbidity (NTU)", "Temperature (°C)", "DO (mg/L)",
    "BOD (mg/L)", "Lead (mg/L)", "Mercury (mg/L)", "Arsenic (mg/L)",
    "Pollution_Level"
]

st.title("Water Pollution Forecast")
st.write(
    "Enter the last 3 readings at this location (oldest first, most recent last), "
    "plus the date/time you\'re forecasting for."
)

default_rows = pd.DataFrame(
    [[7.0, 5.0, 25.0, 5.0, 3.0, 0.005, 0.001, 0.005, 1]] * 3,
    columns=measurement_cols
)
history_df = st.data_editor(default_rows, num_rows="fixed", key="history")

st.subheader("Timestamp to forecast")
forecast_date = st.date_input("Date")
forecast_time = st.time_input("Time")
forecast_dt = datetime.combine(forecast_date, forecast_time)

if st.button("Predict Pollution Level"):
    lag1_values = history_df.iloc[-1]
    roll3_values = history_df[measurement_cols[:-1]].mean()

    input_row = {}
    for col in measurement_cols:
        input_row[f"{col}_lag1"] = lag1_values[col]
    for col in measurement_cols[:-1]:
        input_row[f"{col}_roll3"] = roll3_values[col]

    input_row["hour"] = forecast_dt.hour
    input_row["dayofweek"] = forecast_dt.weekday()

    input_df = pd.DataFrame([input_row])[feature_columns]

    prediction = model.predict(input_df)[0]
    st.success(f"Predicted Pollution Level: {prediction}")

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(input_df)[0]
        proba_df = pd.DataFrame({"Pollution Level": model.classes_, "Probability": proba})
        st.bar_chart(proba_df.set_index("Pollution Level"))
