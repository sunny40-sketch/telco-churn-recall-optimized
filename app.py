import streamlit as st
import pandas as pd
from joblib import dump, load
import os

def train_model():
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import LogisticRegression

    df = pd.read_csv("https://raw.githubusercontent.com/sunny40-sketch/telco-churn-recall-optimized/main/data/WA_Fn-UseC_-Telco-Customer-Churn-checkpoint.csv")
    df["Churn"] = (df["Churn"] == "Yes").astype(int)
    df = df.drop("customerID", axis=1)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = X.select_dtypes(include="object").columns.tolist()

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocessor = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols)
    ])
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(max_iter=4200, class_weight="balanced"))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    pipeline.fit(X_train, y_train)
    dump(pipeline, "churn_model.pkl")
    return pipeline

# load or train
if os.path.exists("churn_model.pkl"):
    loaded_model = load("churn_model.pkl")
else:
    st.info("Training model for first time — please wait...")
    loaded_model = train_model()
    st.success("Model ready!")

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
