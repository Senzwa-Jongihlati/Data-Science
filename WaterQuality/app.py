"""
Water Quality Pollution Level Predictor - Streamlit App
----------------------------------------------------------
A simple web app that takes a single set of water-quality readings
and predicts whether the water is "Polluted" or "Not Polluted",
using the model trained by train_model.py.

Run with:
    streamlit run app.py
"""

import os
import streamlit as st
import pandas as pd
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "pollution_model.joblib")
FEATURE_COLUMNS_PATH = os.path.join(BASE_DIR, "feature_columns.joblib")

st.set_page_config(page_title="Water Pollution Predictor", page_icon="💧")


@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    feature_columns = joblib.load(FEATURE_COLUMNS_PATH)
    return model, feature_columns


model, feature_columns = load_model()

st.title("💧 Water Pollution Level Predictor")
st.write(
    "Enter a water-quality reading below to predict whether the water "
    "sample is **Polluted** or **Not Polluted**."
)

st.subheader("Water Quality Reading")

col1, col2 = st.columns(2)

with col1:
    pH = st.number_input("pH", min_value=0.0, max_value=14.0, value=7.25, step=0.01)
    turbidity = st.number_input(
        "Turbidity (NTU)", min_value=0.0, max_value=50.0, value=10.2, step=0.1
    )
    temperature = st.number_input(
        "Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0, step=0.1
    )
    do = st.number_input(
        "Dissolved Oxygen - DO (mg/L)", min_value=0.0, max_value=20.0, value=5.9, step=0.1
    )

with col2:
    bod = st.number_input(
        "BOD (mg/L)", min_value=0.0, max_value=20.0, value=5.5, step=0.1
    )
    lead = st.number_input(
        "Lead (mg/L)", min_value=0.0, max_value=0.05, value=0.01, step=0.001, format="%.4f"
    )
    mercury = st.number_input(
        "Mercury (mg/L)", min_value=0.0, max_value=0.01, value=0.001, step=0.0001, format="%.5f"
    )
    arsenic = st.number_input(
        "Arsenic (mg/L)", min_value=0.0, max_value=0.05, value=0.01, step=0.001, format="%.4f"
    )

if st.button("Predict Pollution Level", type="primary"):
    input_row = pd.DataFrame(
        [[pH, turbidity, temperature, do, bod, lead, mercury, arsenic]],
        columns=feature_columns,
    )

    prediction = model.predict(input_row)[0]
    label = "🔴 Polluted" if prediction == 1 else "🟢 Not Polluted"

    st.subheader("Result")
    if prediction == 1:
        st.error(f"Prediction: {label}")
    else:
        st.success(f"Prediction: {label}")

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(input_row)[0]
        proba_df = pd.DataFrame(
            {
                "Pollution Level": ["Not Polluted", "Polluted"],
                "Probability": proba,
            }
        ).set_index("Pollution Level")
        st.bar_chart(proba_df)

st.caption(
    "Model: XGBoost classifier trained on historical water-quality readings "
    "(pH, turbidity, temperature, dissolved oxygen, BOD, lead, mercury, arsenic)."
)
st.subheader("What drives the model's predictions")
importance_df = pd.DataFrame({
    "Feature": feature_columns,
    "Importance": model.feature_importances_,
}).sort_values("Importance", ascending=True)
st.bar_chart(importance_df.set_index("Feature"))



st.subheader("Your reading vs. typical values")

# Class averages from the training data (Water_Quality_Dataset.csv)
class_averages = pd.DataFrame({
    "Not Polluted (avg)": [7.29, 5.30, 25.57, 6.68, 3.64, 0.0076, 0.0008, 0.0077],
    "Polluted (avg)":     [7.25, 10.64, 24.92, 5.86, 5.64, 0.0102, 0.0010, 0.0100],
    "Your Reading":       [pH, turbidity, temperature, do, bod, lead, mercury, arsenic],
}, index=feature_columns)

st.dataframe(class_averages.style.format("{:.4f}"))

# Turbidity and BOD are the two strongest signals — chart those specifically
comparison_chart = class_averages.loc[["Turbidity (NTU)", "BOD (mg/L)"]].T
st.bar_chart(comparison_chart)
