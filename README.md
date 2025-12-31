\# Telco Churn Prediction (Recall-Optimized)



\## Goal

Predict customers likely to churn. Business priority: missing a churner (FN) is worse than contacting a non-churner (FP).



\## Model

\- Logistic Regression (class\_weight="balanced")

\- Preprocessing: impute + scale numeric, one-hot encode categoricals (Pipeline + ColumnTransformer)

\- Threshold chosen to achieve recall >= 0.85 and maximize precision under that constraint



\## Run

Place dataset file in:

data/WA\_Fn-UseC\_-Telco-Customer-Churn.csv



Install:

pip install -r requirements.txt



Run:

python src/train.py --data\_path data/WA\_Fn-UseC\_-Telco-Customer-Churn.csv --recall\_target 0.85



