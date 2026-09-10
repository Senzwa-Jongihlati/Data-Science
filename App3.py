import streamlit as st
import pandas as pd
import joblib
import textwrap
import base64
import os
import matplotlib.pyplot as plt
import numpy as np


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Ammoniacal Nitrogen Predictor",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==================================================
# FILE PATHS
# ==================================================

# This must be the model trained in the notebook, which predicts
# AmmoniacalNitrogen_mgL (NOT dissolved oxygen).
MODEL_FILE = "ammoniacal_nitrogen_random_forest.pkl"
FORECAST_MODEL_FILE = "ammoniacal_nitrogen_forecast_model.pkl"

# Point this at your actual raw dataset. It can use either the
# original short column names (date, WT, pH, DO, CODMn, NH3-N, TP,
# TN, EC, Turbidity) or the notebook's renamed versions - both are
# handled automatically by load_data() below.
DATA_FILE = "Dataset_B_Liangtian_Village_Jin_River.csv"

# Maps the raw dataset's original column names to the names the
# notebook renames them to (see notebook cell 10). Applied
# automatically if the raw names are detected.
RAW_TO_RENAMED_COLUMNS = {
    "WT": "WaterTemp_C",
    "DO": "DissolvedOxygen_mgL",
    "CODMn": "PermanganateIndex_mgL",
    "NH3-N": "AmmoniacalNitrogen_mgL",
    "TP": "TotalPhosphorus_mgL",
    "TN": "TotalNitrogen_mgL",
    "EC": "ElectricalConductivity_uScm",
}

# Relative path so the app works on any machine / when deployed.
IMAGE_FILE = os.path.join("assets", "water.jfif")

TARGET_COLUMN = "AmmoniacalNitrogen_mgL"

# Forecasting is built on 4-hour steps (matches the notebook's
# resample('4h') on the raw dataset).
FORECAST_STEP_HOURS = 4
FORECAST_LAGS = [1, 2, 3, 6]


# ==================================================
# FEATURE METADATA (for building input widgets)
# ==================================================
# Keyed by the exact column name the model was trained on. The app
# reads model.feature_names_in_ at runtime and only shows/uses
# whichever of these are actually in the model, so it stays correct
# even if the notebook's feature order or set changes.

FEATURE_METADATA = {
    "WaterTemp_C": dict(
        label="Water Temperature (°C)", group="physical",
        min_value=-10.0, max_value=50.0, default=20.0, step=0.1,
        help="Temperature of the water.", unit="°C"
    ),
    "pH": dict(
        label="pH", group="physical",
        min_value=0.0, max_value=14.0, default=7.0, step=0.1,
        help="Measure of acidity or alkalinity.", unit=""
    ),
    "DissolvedOxygen_mgL": dict(
        label="Dissolved Oxygen (mg/L)", group="physical",
        min_value=0.0, max_value=20.0, default=8.0, step=0.1,
        help="Dissolved oxygen concentration.", unit="mg/L"
    ),
    "Turbidity": dict(
        label="Turbidity", group="physical",
        min_value=0.0, max_value=None, default=1.0, step=0.1,
        help="Measure of water clarity.", unit=""
    ),
    "ElectricalConductivity_uScm": dict(
        label="Electrical Conductivity (µS/cm)", group="physical",
        min_value=0.0, max_value=None, default=100.0, step=1.0,
        help="Electrical conductivity of the water.", unit="µS/cm"
    ),
    "PermanganateIndex_mgL": dict(
        label="Permanganate Index (mg/L)", group="chemical",
        min_value=0.0, max_value=None, default=1.0, step=0.1,
        help="Indicator of organic matter in the water.", unit="mg/L"
    ),
    "TotalPhosphorus_mgL": dict(
        label="Total Phosphorus (mg/L)", group="chemical",
        min_value=0.0, max_value=None, default=0.1, step=0.01,
        help="Total phosphorus concentration.", unit="mg/L"
    ),
    "TotalNitrogen_mgL": dict(
        label="Total Nitrogen (mg/L)", group="chemical",
        min_value=0.0, max_value=None, default=1.0, step=0.1,
        help="Total nitrogen concentration.", unit="mg/L"
    ),
}


# ==================================================
# LOAD MODEL
# ==================================================

@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_FILE):
        return None

    return joblib.load(MODEL_FILE)


model = load_model()

@st.cache_resource
def load_forecast_model():
    if not os.path.exists(FORECAST_MODEL_FILE):
        return None

    return joblib.load(FORECAST_MODEL_FILE)


forecast_model = load_forecast_model()

# image_path = r"C:\Users\USER\Downloads\model - Copy\water.jfif"

# with open(image_path, "rb") as image_file:
#     encoded_image = base64.b64encode(
#         image_file.read()
#     ).decode()

# ==================================================
# CHECK MODEL
# ==================================================

if model is None:

    st.error(
        f"Model file '{MODEL_FILE}' was not found."
    )

    st.info(
        "Run the notebook end-to-end (Restart & Run All) so the "
        "saved .pkl matches the current training code, then place "
        "it in the same folder as this app."
    )

    st.stop()


# ==================================================
# RESOLVE MODEL'S ACTUAL FEATURES
# ==================================================
# Read feature names straight off the model instead of hardcoding
# them, so the app can never silently feed columns in the wrong
# order or set.

if hasattr(model, "feature_names_in_"):
    MODEL_FEATURES = list(model.feature_names_in_)
else:
    st.warning(
        "This model has no feature_names_in_ attribute (it may have "
        "been trained on a plain NumPy array). Falling back to a "
        "hardcoded feature order - double check this matches training."
    )
    MODEL_FEATURES = [
        "WaterTemp_C", "pH", "DissolvedOxygen_mgL",
        "PermanganateIndex_mgL", "TotalPhosphorus_mgL",
        "TotalNitrogen_mgL", "ElectricalConductivity_uScm", "Turbidity"
    ]

UNKNOWN_FEATURES = [f for f in MODEL_FEATURES if f not in FEATURE_METADATA]

if UNKNOWN_FEATURES:
    st.warning(
        "The loaded model expects feature(s) this app doesn't know "
        "how to collect yet: " + ", ".join(UNKNOWN_FEATURES) +
        ". Add them to FEATURE_METADATA in the code."
    )


# ==================================================
# DEFAULT INPUT VALUES (used by number_input + Reset)
# ==================================================

DEFAULTS = {
    feature: FEATURE_METADATA[feature]["default"]
    for feature in MODEL_FEATURES
    if feature in FEATURE_METADATA
}

for _key, _value in DEFAULTS.items():
    if _key not in st.session_state:
        st.session_state[_key] = _value


# ==================================================
# LOAD HISTORICAL DATA
# ==================================================

@st.cache_data
def load_data():

    if not os.path.exists(DATA_FILE):
        return None

    data = pd.read_csv(DATA_FILE)

    # If this is the raw dataset (original short column names),
    # rename it the same way the notebook does, so the rest of the
    # app can always assume the renamed schema.
    columns_to_rename = {
        raw: renamed
        for raw, renamed in RAW_TO_RENAMED_COLUMNS.items()
        if raw in data.columns
    }

    if columns_to_rename:
        data = data.rename(columns=columns_to_rename)

    return data


data = load_data()


# ==================================================
# BACKGROUND IMAGE
# ==================================================
IMAGE_FILE = "water.jfif"
encoded_image = None

if os.path.exists(IMAGE_FILE):
    with open(IMAGE_FILE, "rb") as image_file:
        encoded_image = base64.b64encode(
            image_file.read()
        ).decode()


# ==================================================
# CUSTOM CSS
# ==================================================

if encoded_image:

    background_css = f"""
        background-image:
            linear-gradient(
                rgba(255, 255, 255, 0.45),
                rgba(255, 255, 255, 0.45)
            ),
            url("data:image/jpeg;base64,{encoded_image}");
    """

else:

    background_css = """
        background-color: #f5f7fa;
    """


st.markdown(
    f"""
    <style>

        .stApp {{
            {background_css}

            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}

        .main {{
            padding-top: 1rem;
        }}

        .main-title {{
            font-size: 42px;
            font-weight: 700;
            margin-bottom: 5px;
            color: black;
        }}

        .subtitle {{
            font-size: 18px;
            color: black;
            margin-bottom: 30px;
        }}

        .prediction-card {{
            padding: 30px;
            border-radius: 18px;
            background: linear-gradient(
                135deg,
                #0f4c5c,
                #1b6f8a
            );

            color: white;
            text-align: center;

            margin-top: 20px;
            margin-bottom: 25px;
        }}

        .prediction-label {{
            font-size: 18px;
            opacity: 0.9;
        }}

        .prediction-value {{
            font-size: 48px;
            font-weight: 700;
            margin: 8px 0;
        }}

        .section-title {{
            font-size: 24px;
            font-weight: 600;
            margin-top: 15px;
            margin-bottom: 15px;
            color: black;
        }}

        h1, h2, h3 {{
            color: black !important;
        }}

        .stApp p {{
            color: black;
        }}

        label {{
            color: black !important;
        }}

        div[data-testid="stMetric"] {{
            background-color: rgba(255,255,255,0.75);
            border-radius: 12px;
            padding: 10px;
        }}

        div[data-testid="metric-container"] {{
            border-radius: 12px;
            padding: 10px;
        }}

        .stButton > button {{
            width: 100%;
            border-radius: 10px;
            height: 48px;
            font-size: 17px;
            font-weight: 600;
        }}

    </style>
    """,
    unsafe_allow_html=True
)


# ==================================================
# HEADER
# ==================================================

st.markdown(
    '<div class="main-title">'
    '🌊 Ammoniacal Nitrogen Predictor'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Machine-learning prediction and forecasting of '
    'ammoniacal nitrogen (NH₃-N) from water-quality measurements'
    '</div>',
    unsafe_allow_html=True
)


# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:

    st.header("📊 Model Information")

    st.write(
        "**Model:** Random Forest Regression"
    )

    st.write(
        "**Target:** Ammoniacal Nitrogen (NH₃-N)"
    )

    st.divider()

    st.subheader("Model Performance")

    # Values from the notebook's final tuned Random Forest evaluation
    # (test set). Re-run the notebook and update these if you retrain.
    st.metric(
        "R² Score",
        "0.8669"
    )

    st.metric(
        "MAE",
        "0.0132 mg/L"
    )

    st.metric(
        "RMSE",
        "0.0212 mg/L"
    )

    st.divider()

    st.caption(
        "The model was trained using historical "
        "water-quality observations from 2021-2024."
    )


# ==================================================
# INPUT SECTION
# ==================================================

st.markdown(
    '<div class="section-title">'
    '💧 Water Quality Measurements'
    '</div>',
    unsafe_allow_html=True
)

st.write(
    "Enter the available water-quality measurements below. The "
    "Random Forest model will use these values to estimate "
    "ammoniacal nitrogen (NH₃-N)."
)


# ==================================================
# INPUT COLUMNS (built dynamically from the model's own features)
# ==================================================

col1, col2 = st.columns(2)

physical_features = [
    f for f in MODEL_FEATURES
    if FEATURE_METADATA.get(f, {}).get("group") == "physical"
]

chemical_features = [
    f for f in MODEL_FEATURES
    if FEATURE_METADATA.get(f, {}).get("group") == "chemical"
]

input_values = {}

with col1:

    st.subheader("🌡️ Physical Conditions")

    for feature in physical_features:

        meta = FEATURE_METADATA[feature]

        input_values[feature] = st.number_input(
            meta["label"],
            min_value=meta["min_value"],
            max_value=meta["max_value"],
            step=meta["step"],
            key=feature,
            help=meta["help"]
        )

with col2:

    st.subheader("🧪 Chemical Measurements")

    for feature in chemical_features:

        meta = FEATURE_METADATA[feature]

        input_values[feature] = st.number_input(
            meta["label"],
            min_value=meta["min_value"],
            max_value=meta["max_value"],
            step=meta["step"],
            key=feature,
            help=meta["help"]
        )


# ==================================================
# CREATE INPUT DATA (columns in the exact order the model expects)
# ==================================================

input_data = pd.DataFrame(
    [[input_values[feature] for feature in MODEL_FEATURES]],
    columns=MODEL_FEATURES
)


# ==================================================
# BUTTONS
# ==================================================

st.divider()

predict_col, forecast_col, reset_col = st.columns(
    [2, 2, 1]
)


with predict_col:

    predict_button = st.button(
        "🔮 Predict Ammoniacal Nitrogen",
        type="primary"
    )


with forecast_col:

    forecast_button = st.button(
        "📈 Forecast Future NH₃-N"
    )


with reset_col:

    reset_button = st.button(
        "↻ Reset"
    )


# ==================================================
# RESET
# ==================================================

def reset_defaults():
    for _key, _value in DEFAULTS.items():
        st.session_state[_key] = _value




# ==================================================
# PREDICTION
# ==================================================

if predict_button:

    try:

        prediction = model.predict(
            input_data
        )[0]

        prediction = float(prediction)

        # ------------------------------------------
        # PREDICTION CARD
        # ------------------------------------------

        st.markdown(
            '<div class="prediction-label">Predicted Ammoniacal Nitrogen (NH₃-N)</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="prediction-value">{prediction:.4f} mg/L</div>',
            unsafe_allow_html=True
        )

        # ------------------------------------------
        # INTERPRETATION
        # ------------------------------------------

        explanation = (
            f"The Random Forest model predicts an ammoniacal "
            f"nitrogen concentration of {prediction:.4f} mg/L "
            f"based on the water-quality measurements provided."
        )

        st.info(explanation)


        # ------------------------------------------
        # WATER QUALITY STATUS
        # ------------------------------------------
        # Rough tiers loosely based on common surface-water NH3-N
        # guideline bands (lower = better). These are indicative
        # only, not a regulatory classification - adjust to whatever
        # standard applies to your monitoring site.

        st.subheader(
            "💧 Ammoniacal Nitrogen Status"
        )

        if prediction <= 0.15:

            status = "Good"

        elif prediction <= 0.5:

            status = "Moderate"

        else:

            status = "Elevated"


        status_col1, status_col2 = st.columns(2)

        with status_col1:

            st.metric(
                "Predicted NH₃-N",
                f"{prediction:.4f} mg/L"
            )

        with status_col2:

            st.metric(
                "Status",
                status
            )

        st.caption(
            "Status bands are an indicative guideline, not a "
            "regulatory classification."
        )


        # ------------------------------------------
        # INPUT SUMMARY
        # ------------------------------------------

        st.subheader(
            "📋 Prediction Input Summary"
        )

        display_data = pd.DataFrame({
            "Measurement": [
                FEATURE_METADATA[f]["label"] for f in MODEL_FEATURES
            ],
            "Value": [
                f"{input_values[f]:.4f} {FEATURE_METADATA[f]['unit']}".strip()
                for f in MODEL_FEATURES
            ]
        })

        st.dataframe(
            display_data,
            use_container_width=True,
            hide_index=True
        )


    except Exception as e:

        st.error(
            f"Prediction error: {e}"
        )


# ==================================================
# HISTORICAL NH3-N TREND
# ==================================================

st.divider()

st.subheader(
    "📈 Historical Ammoniacal Nitrogen"
)


if data is None:

    st.warning(
        f"Dataset '{DATA_FILE}' was not found."
    )

    st.info(
        "Place your CSV file in the same folder as this app, "
        "with columns matching the notebook's renamed schema."
    )

else:

    required_columns = [
        "date",
        TARGET_COLUMN
    ]

    missing_columns = [
        column for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:

        st.error(
            "The CSV file is missing these columns: "
            + ", ".join(missing_columns)
        )

    else:

        data["date"] = pd.to_datetime(
            data["date"],
            errors="coerce"
        )

        data[TARGET_COLUMN] = pd.to_numeric(
            data[TARGET_COLUMN],
            errors="coerce"
        )

        historical_data = data.dropna(
            subset=["date", TARGET_COLUMN]
        ).copy()

        historical_data = historical_data.sort_values("date")

        if len(historical_data) > 0:

            fig, ax = plt.subplots(figsize=(12, 5))

            ax.plot(
                historical_data["date"],
                historical_data[TARGET_COLUMN],
                linewidth=2
            )

            ax.set_xlabel("Date")
            ax.set_ylabel("Ammoniacal Nitrogen (mg/L)")
            ax.set_title("Ammoniacal Nitrogen Trend")

            plt.xticks(rotation=45)
            plt.tight_layout()

            st.pyplot(fig)
            plt.close(fig)

        else:

            st.warning(
                "No valid date and NH₃-N values were found in "
                "the dataset."
            )


# ==================================================
# FUTURE NH3-N FORECAST
# ==================================================

st.divider()

st.subheader(
    "🔮 Future Ammoniacal Nitrogen Forecast"
)

# The forecast model is already trained and loaded from the .pkl file.
# Therefore, this section DOES NOT retrain a Random Forest.

if data is None:
    st.info(
        f"Load a dataset containing 'date', '{TARGET_COLUMN}', "
        "and the model's input columns to enable future forecasting."
    )

elif TARGET_COLUMN not in data.columns:
    st.error(
        f"The dataset does not contain the target column "
        f"'{TARGET_COLUMN}'."
    )

elif "date" not in data.columns:
    st.info(
        "Add a 'date' column to the dataset to enable forecasting."
    )

elif forecast_model is None:
    st.error(
        f"Forecast model file '{FORECAST_MODEL_FILE}' was not found."
    )
    st.info(
        "Place the trained forecast .pkl file in the same folder "
        "as this Streamlit app."
    )

else:
    # --------------------------------------------------
    # GET THE FEATURES EXPECTED BY THE SAVED FORECAST MODEL
    # --------------------------------------------------
    if hasattr(forecast_model, "feature_names_in_"):
        FORECAST_MODEL_FEATURES = list(
            forecast_model.feature_names_in_
        )
    else:
        # Fallback only if the saved model was trained without
        # column names.
        FORECAST_MODEL_FEATURES = (
            MODEL_FEATURES
            + [f"NH3_lag_{lag}" for lag in FORECAST_LAGS]
        )

    lag_columns = [
        f"NH3_lag_{lag}"
        for lag in FORECAST_LAGS
    ]

    exogenous_features = [
        feature
        for feature in FORECAST_MODEL_FEATURES
        if feature not in lag_columns
    ]

    missing_exogenous = [
        feature
        for feature in exogenous_features
        if feature not in data.columns
    ]

    missing_lags = [
        feature
        for feature in lag_columns
        if feature not in FORECAST_MODEL_FEATURES
    ]

    if missing_exogenous:
        st.warning(
            "The dataset is missing these forecast input columns: "
            + ", ".join(missing_exogenous)
        )

    elif missing_lags:
        st.warning(
            "The saved forecast model does not contain all required "
            "NH₃ lag features: " + ", ".join(missing_lags)
        )

    else:
        # --------------------------------------------------
        # CLEAN DATA
        # --------------------------------------------------
        forecast_data = data.copy()

        forecast_data["date"] = pd.to_datetime(
            forecast_data["date"],
            errors="coerce"
        )

        numeric_columns = [
            TARGET_COLUMN
        ] + exogenous_features

        for column in numeric_columns:
            forecast_data[column] = pd.to_numeric(
                forecast_data[column],
                errors="coerce"
            )

        forecast_data = (
            forecast_data
            .dropna(
                subset=["date"] + numeric_columns
            )
            .sort_values("date")
            .reset_index(drop=True)
        )

        # --------------------------------------------------
        # YEAR SELECTOR
        # --------------------------------------------------
        forecast_years = st.slider(
            "Number of years to forecast",
            min_value=1,
            max_value=5,
            value=1,
            step=1,
            help=(
                f"Forecasts use {FORECAST_STEP_HOURS}-hour steps "
                "and the saved forecast model."
            )
        )

        forecast_days = forecast_years * 365

        forecast_steps = int(
            forecast_days * 24 / FORECAST_STEP_HOURS
        )

        if forecast_button:

            min_required = max(FORECAST_LAGS) + 10

            if len(forecast_data) < min_required:
                st.error(
                    f"At least {min_required} historical observations "
                    "are required for forecasting."
                )

            else:
                try:
                    # --------------------------------------------------
                    # RESAMPLE HISTORICAL DATA TO 4-HOUR INTERVALS
                    # --------------------------------------------------
                    resampled = (
                        forecast_data
                        .set_index("date")[
                            [TARGET_COLUMN] + exogenous_features
                        ]
                        .resample(f"{FORECAST_STEP_HOURS}h")
                        .mean()
                        .interpolate(method="time")
                        .ffill()
                        .bfill()
                        .reset_index()
                    )

                    if len(resampled) < min_required:
                        st.error(
                            "There are not enough valid observations after "
                            "resampling the historical data."
                        )
                        st.stop()

                    # --------------------------------------------------
                    # CREATE NH3 LAG FEATURES
                    # --------------------------------------------------
                    for lag in FORECAST_LAGS:
                        resampled[f"NH3_lag_{lag}"] = (
                            resampled[TARGET_COLUMN].shift(lag)
                        )

                    # Remove rows that cannot have lag values.
                    forecasting_df = resampled.dropna(
                        subset=lag_columns
                    ).reset_index(drop=True)

                    if forecasting_df.empty:
                        st.error(
                            "Could not create the required NH₃ lag features."
                        )
                        st.stop()

                    # --------------------------------------------------
                    # CREATE MONTHLY SEASONAL PROFILE
                    # --------------------------------------------------
                    resampled["month"] = (
                        resampled["date"].dt.month
                    )

                    seasonal_profile = (
                        resampled
                        .groupby("month")[exogenous_features]
                        .mean()
                    )

                    overall_exogenous_mean = (
                        resampled[exogenous_features].mean()
                    )

                    # --------------------------------------------------
                    # INITIAL NH3 HISTORY
                    # --------------------------------------------------
                    max_lag = max(FORECAST_LAGS)

                    initial_history = (
                        resampled[TARGET_COLUMN]
                        .astype(float)
                        .to_numpy(dtype=np.float64)
                    )

                    if len(initial_history) < max_lag:
                        st.error(
                            "Not enough historical NH₃-N values to create "
                            "the required lag features."
                        )
                        st.stop()

                    # --------------------------------------------------
                    # FUTURE DATES AND SEASONAL EXOGENOUS VALUES
                    # --------------------------------------------------
                    last_date = resampled["date"].iloc[-1]

                    future_dates = pd.date_range(
                        start=last_date + pd.Timedelta(
                            hours=FORECAST_STEP_HOURS
                        ),
                        periods=forecast_steps,
                        freq=f"{FORECAST_STEP_HOURS}h"
                    )

                    future_months = future_dates.month.to_numpy()

                    seasonal_lookup = seasonal_profile.reindex(
                        future_months
                    )

                    seasonal_lookup = seasonal_lookup.fillna(
                        overall_exogenous_mean
                    )

                    future_exogenous = seasonal_lookup[
                        exogenous_features
                    ].to_numpy(dtype=np.float64)

                    # --------------------------------------------------
                    # FAST RECURSIVE FORECAST USING SAVED .PKL
                    # --------------------------------------------------
                    # The model is NOT retrained here.
                    # We traverse the already-trained Random Forest trees
                    # directly instead of calling sklearn predict() once
                    # for every 4-hour step. This keeps the same model and
                    # prediction logic while removing the main bottleneck.
                    trees = getattr(forecast_model, "estimators_", None)

                    if trees is None or len(trees) == 0:
                        raise ValueError(
                            "The saved forecast model is not a trained "
                            "Random Forest model."
                        )

                    feature_positions = {
                        feature: index
                        for index, feature in enumerate(
                            FORECAST_MODEL_FEATURES
                        )
                    }

                    exogenous_positions = {
                        feature: index
                        for index, feature in enumerate(
                            exogenous_features
                        )
                    }

                    lag_positions = {
                        lag: feature_positions[f"NH3_lag_{lag}"]
                        for lag in FORECAST_LAGS
                    }

                    tree_structures = []

                    for estimator in trees:
                        tree = estimator.tree_
                        tree_structures.append((
                            tree.feature,
                            tree.threshold,
                            tree.children_left,
                            tree.children_right,
                            tree.value[:, 0, 0]
                        ))

                    n_steps = len(future_exogenous)
                    history = np.empty(
                        len(initial_history) + n_steps,
                        dtype=np.float64
                    )
                    history[:len(initial_history)] = initial_history

                    future_predictions = np.empty(
                        n_steps,
                        dtype=np.float64
                    )

                    model_input = np.empty(
                        len(FORECAST_MODEL_FEATURES),
                        dtype=np.float64
                    )

                    history_length = len(initial_history)
                    number_of_trees = len(tree_structures)

                    progress_bar = st.progress(0)

                    for step in range(n_steps):

                        for feature in exogenous_features:
                            model_input[
                                feature_positions[feature]
                            ] = future_exogenous[
                                step,
                                exogenous_positions[feature]
                            ]

                        for lag in FORECAST_LAGS:
                            model_input[
                                lag_positions[lag]
                            ] = history[
                                history_length - lag
                            ]

                        tree_sum = 0.0

                        for (
                            tree_features,
                            tree_thresholds,
                            tree_left,
                            tree_right,
                            tree_values
                        ) in tree_structures:

                            node = 0

                            while tree_left[node] != tree_right[node]:
                                feature_index = tree_features[node]

                                if (
                                    model_input[feature_index]
                                    <= tree_thresholds[node]
                                ):
                                    node = tree_left[node]
                                else:
                                    node = tree_right[node]

                            tree_sum += tree_values[node]

                        predicted_nh3 = (
                            tree_sum / number_of_trees
                        )

                        # NH3-N concentration cannot be negative.
                        predicted_nh3 = max(0.0, predicted_nh3)

                        future_predictions[step] = predicted_nh3
                        history[history_length] = predicted_nh3
                        history_length += 1

                        if (
                            step == n_steps - 1
                            or (step + 1) % max(1, n_steps // 100) == 0
                        ):
                            progress_bar.progress(
                                min((step + 1) / n_steps, 1.0)
                            )

                    progress_bar.empty()
                    # --------------------------------------------------
                    # CREATE FORECAST DATAFRAME
                    # --------------------------------------------------
                    future_df = pd.DataFrame({
                        "Date": future_dates,
                        "Forecast_NH3_mgL": future_predictions
                    })

                    # --------------------------------------------------
                    # SUCCESS MESSAGE
                    # --------------------------------------------------
                    st.success(
                        f"NH₃-N forecast generated for the next "
                        f"{forecast_years} year"
                        f"{'s' if forecast_years != 1 else ''} "
                        f"({forecast_steps} steps of "
                        f"{FORECAST_STEP_HOURS} hours each)."
                    )

                    st.caption(
                        f"Using saved forecast model: "
                        f"{FORECAST_MODEL_FILE}"
                    )

                    st.caption(
                        "⚠️ Forecasting multiple years ahead compounds "
                        "uncertainty. Treat long-term values as a trend "
                        "indicator rather than precise measurements."
                    )

                    st.caption(
                        "Future temperature, pH and other exogenous "
                        "inputs are estimated from historical monthly "
                        "averages."
                    )

                    # --------------------------------------------------
                    # FORECAST GRAPH
                    # --------------------------------------------------
                    fig3, ax3 = plt.subplots(
                        figsize=(12, 5)
                    )

                    ax3.plot(
                        resampled["date"],
                        resampled[TARGET_COLUMN],
                        label="Historical NH₃-N",
                        linewidth=2
                    )

                    ax3.plot(
                        future_df["Date"],
                        future_df["Forecast_NH3_mgL"],
                        linestyle="--",
                        linewidth=2,
                        label="Forecast NH₃-N"
                    )

                    ax3.set_xlabel("Date")
                    ax3.set_ylabel(
                        "Ammoniacal Nitrogen (mg/L)"
                    )
                    ax3.set_title(
                        "Historical and Future NH₃-N Forecast"
                    )
                    ax3.legend()

                    plt.xticks(rotation=45)
                    plt.tight_layout()

                    st.pyplot(fig3)
                    plt.close(fig3)

                    # --------------------------------------------------
                    # FORECAST TABLE
                    # --------------------------------------------------
                    st.subheader(
                        "📋 Forecast Values"
                    )

                    display_forecast = future_df.copy()

                    display_forecast[
                        "Forecast_NH3_mgL"
                    ] = (
                        display_forecast[
                            "Forecast_NH3_mgL"
                        ].round(4)
                    )

                    st.dataframe(
                        display_forecast,
                        use_container_width=True,
                        hide_index=True
                    )

                    # --------------------------------------------------
                    # FORECAST SUMMARY
                    # --------------------------------------------------
                    average_forecast = np.mean(
                        future_predictions
                    )

                    minimum_forecast = np.min(
                        future_predictions
                    )

                    maximum_forecast = np.max(
                        future_predictions
                    )

                    summary_col1, summary_col2, summary_col3 = (
                        st.columns(3)
                    )

                    with summary_col1:
                        st.metric(
                            "Average Forecast NH₃-N",
                            f"{average_forecast:.4f} mg/L"
                        )

                    with summary_col2:
                        st.metric(
                            "Minimum Forecast NH₃-N",
                            f"{minimum_forecast:.4f} mg/L"
                        )

                    with summary_col3:
                        st.metric(
                            "Maximum Forecast NH₃-N",
                            f"{maximum_forecast:.4f} mg/L"
                        )

                except Exception as e:
                    st.error(
                        f"Forecasting error: {e}"
                    )


# ==================================================
# FEATURE IMPORTANCE
# ==================================================

st.divider()

st.subheader(
    "📊 Feature Importance"
)

try:

    importance = pd.DataFrame({
        "Feature": MODEL_FEATURES,
        "Importance": model.feature_importances_
    })

    importance = importance.sort_values(
        "Importance", ascending=False
    )

    st.dataframe(
        importance,
        use_container_width=True,
        hide_index=True
    )

    fig4, ax4 = plt.subplots(figsize=(10, 5))

    ax4.barh(
        importance["Feature"],
        importance["Importance"]
    )

    ax4.set_xlabel("Importance")
    ax4.set_ylabel("Feature")
    ax4.set_title("Features Affecting Ammoniacal Nitrogen")

    ax4.invert_yaxis()
    plt.tight_layout()

    st.pyplot(fig4)
    plt.close(fig4)

except Exception as e:

    st.warning(
        f"Could not display feature importance: {e}"
    )


# ==================================================
# FOOTER
# ==================================================

st.divider()

st.caption(
    "Ammoniacal Nitrogen Prediction System • "
    "Random Forest Regression • "
    "Historical and Future Forecasting"
)
