import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os

st.set_page_config(
    page_title="Customer Churn Predictor",
    layout="wide"
)

st.title("Customer Churn Prediction System")
st.markdown("Predict whether a customer will churn or not")

st.sidebar.header("Enter Customer Data:")

age = st.sidebar.slider("Age", 10, 100, 30)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
tenure = st.sidebar.slider("Tenure (Months)", 0, 130, 10)
monthly_charges = st.sidebar.slider("Monthly Charges", 30, 150, 80)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
scaler_path = os.path.join(BASE_DIR, "scaler.pkl")
model_path = os.path.join(BASE_DIR, "model.pkl")
scaler = joblib.load(scaler_path)
model = joblib.load(model_path)

gender_selected = 1 if gender == "Female" else 0

X = np.array([[age, gender_selected, tenure, monthly_charges]])
X_scaled = scaler.transform(X)

prediction = model.predict(X_scaled)[0]
probability = model.predict_proba(X_scaled)[0][1]
if prediction == 1:
    st.error("Customer will Churn")
else:
    st.success("Customer will Stay")
st.metric("Churn Probability", f"{probability*100:.2f}%")

st.subheader("Model Information")
st.write("""
Model Used: Support Vector Machine (SVM)  
Features:
- Age
- Gender
- Tenure
- Monthly Charges
""")

data = pd.DataFrame({
    "Feature": ["Age", "Tenure", "MonthlyCharges"],
    "Value": [age, tenure, monthly_charges]
})

st.bar_chart(data.set_index("Feature"))