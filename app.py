import streamlit as st
import pickle
import numpy as np

# -----------------------------
# Load Trained Model
# -----------------------------
try:
    model = pickle.load(open("diabetes_model.pkl", "rb"))
except FileNotFoundError:
    st.error("PKL model file not found")
    st.stop()

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Diabetes Prediction")

st.title("🩺 Diabetes Prediction App")

st.write("Enter patient details below")

# User Inputs
gender = st.selectbox(
    "Gender",
    [0, 1]
)

age = st.number_input(
    "Age",
    min_value=1,
    max_value=100,
    value=25
)

hypertension = st.selectbox(
    "Hypertension",
    [0, 1]
)

heart_disease = st.selectbox(
    "Heart Disease",
    [0, 1]
)

smoking_history = st.selectbox(
    "Smoking History",
    [0, 1, 2, 3]
)

bmi = st.number_input(
    "BMI",
    min_value=10.0,
    max_value=50.0,
    value=22.0
)

hba1c = st.number_input(
    "HbA1c Level",
    min_value=1.0,
    max_value=15.0,
    value=5.0
)

glucose = st.number_input(
    "Blood Glucose Level",
    min_value=50,
    max_value=300,
    value=120
)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict"):

    input_data = np.array([[
        gender,
        age,
        hypertension,
        heart_disease,
        smoking_history,
        bmi,
        hba1c,
        glucose
    ]])

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("⚠️ Diabetes Detected")
    else:
        st.success("✅ No Diabetes")
