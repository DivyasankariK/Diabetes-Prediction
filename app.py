import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

# -----------------------------
# Load Dataset
# -----------------------------
diabetes = pd.read_csv("diabetes_prediction_dataset.csv")

# -----------------------------
# Encode Categorical Columns
# -----------------------------
le_gender = LabelEncoder()
le_smoke = LabelEncoder()

diabetes['gender'] = le_gender.fit_transform(
    diabetes['gender'].astype(str)
)

diabetes['smoking_history'] = le_smoke.fit_transform(
    diabetes['smoking_history'].astype(str)
)

# -----------------------------
# Features and Target
# -----------------------------
X = diabetes[['gender',
              'age',
              'hypertension',
              'heart_disease',
              'smoking_history',
              'bmi',
              'HbA1c_level',
              'blood_glucose_level']]

y = diabetes['diabetes']

# -----------------------------
# Train Model
# -----------------------------
model = LinearRegression()
model.fit(X, y)

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Diabetes Prediction")

st.title("🩺 Diabetes Prediction App")

st.write("Enter patient details below")

# User Inputs
gender = st.selectbox(
    "Gender",
    ["Female", "Male"]
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

smoking = st.selectbox(
    "Smoking History",
    ["never", "former", "current", "not current"]
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

    # Encode Inputs
    gender_val = le_gender.transform([gender])[0]
    smoke_val = le_smoke.transform([smoking])[0]

    # Prediction
    prediction = model.predict([[
        gender_val,
        age,
        hypertension,
        heart_disease,
        smoke_val,
        bmi,
        hba1c,
        glucose
    ]])

    # Result
    if prediction[0] >= 0.5:
        st.error("⚠️ Diabetes Detected")
    else:
        st.success("✅ No Diabetes")
