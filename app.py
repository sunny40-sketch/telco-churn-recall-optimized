import streamlit as st
import pandas as pd
from joblib import dump, load

loaded_model = load(r"C:\Users\kandu\OneDrive\Pictures\Desktop\Chrun_project\churn_model.pkl")
st.title("Customer Churn Prediction")
st.write("Enter customer details to predict churn.")

gender = st.selectbox("Gender", ["Male", "Female"])
senior_citizen = st.selectbox("Senior Citizen", [0, 1])
partner = st.selectbox("Partner", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["Yes", "No"])
phone_service = st.selectbox("Phone Service", ["Yes", "No"])
multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
monthly_charges = st.number_input("Monthly Charges", min_value=0.0, max_value=200.0, value=50.0)
total_charges = st.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=600.0)
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
payment_method = st.selectbox("Payment Method", [
    "Electronic check", "Mailed check",
    "Bank transfer (automatic)", "Credit card (automatic)"
])

if st.button("Predict Churn"):
    input_data = pd.DataFrame([[
        gender, senior_citizen, partner, dependents,
        tenure, phone_service, multiple_lines, internet_service,
        online_security, online_backup, device_protection,
        tech_support, streaming_tv, streaming_movies,
        contract, paperless_billing, payment_method,
        monthly_charges, total_charges
    ]], columns=[
        "gender", "SeniorCitizen", "Partner", "Dependents",
        "tenure", "PhoneService", "MultipleLines", "InternetService",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies",
        "Contract", "PaperlessBilling", "PaymentMethod",
        "MonthlyCharges", "TotalCharges"
    ])

    probability = loaded_model.predict_proba(input_data)[0][1]
    prediction = loaded_model.predict(input_data)[0]

    if prediction == 1:
        st.error(f"High churn risk — {probability:.0%} probability")
    else:
        st.success(f"Low churn risk — {probability:.0%} probability")