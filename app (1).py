import streamlit as st
import joblib
import numpy as np

# Load the saved model
model = joblib.load("insurance_model.pkl")

# App title
st.set_page_config(page_title="Health Insurance Cost Predictor", layout="centered")
st.title("💰 Health Insurance Cost Predictor")

st.markdown("Enter your details below to estimate your insurance charges.")

# --- User Inputs ---
age = st.slider("Age", 18, 100, 30)
sex = st.radio("Sex", ["Male", "Female"])
bmi = st.slider("BMI", 10.0, 50.0, 25.0)
children = st.selectbox("Number of Children", [0, 1, 2, 3, 4, 5])
smoker = st.radio("Do you smoke?", ["Yes", "No"])
region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

# --- Preprocessing ---
sex_val = 1 if sex == "Male" else 0
smoker_val = 1 if smoker == "Yes" else 0
region_vals = {
    "northeast": [0, 0, 0],
    "northwest": [1, 0, 0],
    "southeast": [0, 1, 0],
    "southwest": [0, 0, 1]
}
region_encoded = region_vals[region]

# Final feature array
features = np.array([[age, sex_val, bmi, children, smoker_val] + region_encoded])

# --- Prediction ---
if st.button("Predict Insurance Cost"):
    prediction = model.predict(features)[0]
    st.success(f"💸 Estimated Insurance Cost: ₹{prediction:,.2f}")
