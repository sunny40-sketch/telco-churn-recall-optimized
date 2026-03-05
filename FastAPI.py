from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI()
model = joblib.load(r"C:\Users\kandu\OneDrive\Pictures\Desktop\Chrun_project\artifacts\telco_churn_pipeline.pkl")

class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float
    
@app.get("/")
def home():
    return {"message" : "churn Prediction API is running"}


@app.post("/predict/")
def predict(data: CustomerData):
    # Convert the input data to a DataFrame
    input_data = pd.DataFrame([data.dict()])
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    return {
        "prediction": int(prediction),
        "probability": round(float(probability), 2),
        "result": "Churn" if prediction == 1 else "No Churn"
    }