# Telco Customer Churn Prediction System

This project builds an **end-to-end machine learning system** to predict customer churn.

It includes:

• A **training pipeline** for model development
• A **REST API** for model inference
• A **Streamlit web app** for interactive predictions

The goal is to identify customers likely to churn so businesses can take proactive retention actions.

---

# Project Overview

Customer churn prediction is a **binary classification problem** where the model predicts whether a customer will leave the service.

Target variable:

* **1 → Customer churns**
* **0 → Customer stays**

The system prioritizes **recall**, ensuring that most churn-risk customers are detected.

---

# System Architecture

Training Pipeline → Saved Model → API → Streamlit UI

1. **Training Script**

   * Builds preprocessing pipeline
   * Trains Logistic Regression model
   * Finds recall-optimized decision threshold
   * Saves model and configuration

2. **Inference API**

   * Loads trained model
   * Accepts customer data
   * Returns churn probability and prediction

3. **Streamlit App**

   * User interface for entering customer details
   * Calls the model to generate predictions
   * Displays churn risk interactively

---

<<<<<<< HEAD
# Machine Learning Pipeline

### Feature Processing

Numerical Features

* Median imputation
* Standard scaling

Categorical Features

* Missing value imputation
* One-hot encoding

### Model

Logistic Regression with

```
class_weight="balanced"
```

to address class imbalance.

---

# Threshold Optimization

Instead of using the default **0.5 classification threshold**, the system searches for a threshold that achieves a **target recall**.

Process:

1. Generate prediction probabilities
2. Evaluate thresholds from **0.05 → 0.95**
3. Choose the threshold that

• meets the recall target
• maximizes precision

This creates a business-optimized decision point.

---

# Evaluation Metrics

The model reports:

* Precision
* Recall
* F1 Score
* Confusion Matrix
* Classification Report

Two evaluation modes are shown:

Baseline (threshold = 0.5)

Optimized threshold (recall-target based)

---

# Saved Artifacts

After training the following files are created.

Model pipeline

```
artifacts/telco_churn_pipeline.pkl
```

Contains:

* preprocessing pipeline
* trained model

Threshold configuration

```
artifacts/telco_threshold_config.json
```

Contains:

* recall target
* selected threshold
* evaluation metrics
* expected input features

---

# Project Structure

```
churn-project
│
├── train.py
│
├── api
│   └── app.py
│
├── streamlit_app
│   └── app.py
│
├── artifacts
│   ├── telco_churn_pipeline.pkl
│   └── telco_threshold_config.json
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# Installation

Clone the repository

```
Install dependencies

```
pip install -r requirements.txt
```

---

# Train the Model

```
python train.py --data_path "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
```

Artifacts will be saved to

```
artifacts/
```

---

# Run the API

Start the API server

```
python api/app.py
```

Example request:

```
POST /predict
```

Returns

```
{
 "churn_probability": 0.82,
 "prediction": 1
}
```

---

# Run the Streamlit App

```
streamlit run streamlit_app/app.py
```

The app will open in your browser where you can enter customer data and view predictions.

---

# Dataset

Telco Customer Churn Dataset

Features include:

* customer tenure
* contract type
* payment method
* internet service
* monthly charges
* total charges

Target:

```
Churn
```

---

# Future Improvements

Possible improvements include

• Random Forest / XGBoost models
• Hyperparameter tuning
• MLflow experiment tracking
• Docker deployment
• CI/CD pipeline for model updates

---

# Author

Sunny Kumar Kandula

Master's in Artificial Intelligence (Machine Learning)

Interested in:

Machine Learning Engineering
AI Systems
Data Engineering
