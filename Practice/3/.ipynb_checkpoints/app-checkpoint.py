import streamlit as st
import joblib
import pandas as pd

@st.cache_resource
def load_model():
    return joblib.load("Iris_Logistic_Regression.joblib")

model = load_model()

st.title("Iris Species Classifier")

sepal_length = st.slider("Sepal length (cm)", 4.3, 7.9, 5.8, 0.1)
sepal_width = st.slider("Sepal width (cm)", 2.0, 4.4, 3.0, 0.1)
petal_length = st.slider("Petal length (cm)", 1.0, 6.9, 3.8, 0.1)
petal_width = st.slider("Petal width (cm)", 0.1, 2.5, 1.2, 0.1)

X = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                  columns=["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"])

prediction = model.predict(X)[0]
st.subheader(f"Predicted species: {prediction}")
