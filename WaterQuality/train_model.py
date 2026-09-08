"""
Water Quality Pollution Level Classifier - Training Script
------------------------------------------------------------
This script contains only the data-cleaning and classification steps
from the original notebook (everything up to, but not including, the
forecasting section). It trains the best-performing model (XGBoost,
~99% test accuracy in the original notebook) and saves it so the
Streamlit app (app.py) can load it for live predictions.

Run this once, from the same folder as Water_Quality_Dataset.csv:
    python train_model.py

It produces two files used by app.py:
    pollution_model.joblib      - the trained XGBoost model
    feature_columns.joblib      - the exact feature order the model expects
"""

import pandas as pd
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import joblib

DATA_PATH = "Water_Quality_Dataset.csv"
MODEL_PATH = "pollution_model.joblib"
FEATURE_COLUMNS_PATH = "feature_columns.joblib"

FEATURE_COLUMNS = [
    "pH",
    "Turbidity (NTU)",
    "Temperature (°C)",
    "DO (mg/L)",
    "BOD (mg/L)",
    "Lead (mg/L)",
    "Mercury (mg/L)",
    "Arsenic (mg/L)",
]


def load_and_clean_data(path: str) -> pd.DataFrame:
    """Load the raw CSV and collapse the 3-class Pollution_Level into
    a binary target: 0 = Not Polluted (original levels 0 and 1),
    1 = Polluted (original level 2)."""
    df = pd.read_csv(path)

    df["Pollution_Level"] = df["Pollution_Level"].replace({0: 1, 1: 0, 2: 1})

    return df


def build_features(df: pd.DataFrame):
    """Split the cleaned dataframe into features (X) and target (y)."""
    y = df["Pollution_Level"]
    X = df[FEATURE_COLUMNS]
    return X, y


def train_model(X, y):
    """Stratified train/test split, SMOTE oversampling on the training
    set only, then fit an XGBoost classifier."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Before SMOTE:", Counter(y_train))
    min_class_count = min(Counter(y_train).values())
    k = min(5, max(1, min_class_count - 1))
    smote = SMOTE(random_state=42, k_neighbors=k)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
    print("After SMOTE:", Counter(y_train_sm))

    model = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=42,
    )
    model.fit(X_train_sm, y_train_sm)

    y_pred = model.predict(X_test)
    print("\nTest set performance:")
    print(classification_report(y_test, y_pred))

    return model


def main():
    df = load_and_clean_data(DATA_PATH)
    X, y = build_features(df)
    model = train_model(X, y)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(FEATURE_COLUMNS, FEATURE_COLUMNS_PATH)
    print(f"\nSaved model to {MODEL_PATH}")
    print(f"Saved feature column order to {FEATURE_COLUMNS_PATH}")


if __name__ == "__main__":
    main()
