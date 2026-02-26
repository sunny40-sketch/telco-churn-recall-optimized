# Telco Churn Prediction (Recall-Optimized)

## Results (Test Set)

- **Recall (Churn):** 0.85  
- **Precision (Churn):** 0.49  
- **F1-score:** 0.62  
- **Decision threshold:** 0.43  
- **Missed churners (FN):** 70  
- **Customers flagged for retention:** 815  

The decision threshold was selected to guarantee high recall, minimizing missed churners as per business requirements.

---

## Goal

Predict customers likely to churn.  
Business priority: **missing a churner (FN) is worse than contacting a non-churner (FP).**

---

## Model

- Logistic Regression (`class_weight="balanced"`)
- Preprocessing:
  - Numeric features: median imputation + standard scaling
  - Categorical features: most-frequent imputation + one-hot encoding  
  - Implemented using `Pipeline` and `ColumnTransformer`
- Threshold selected to achieve **recall ≥ 0.85**, then maximize precision under that constraint

---

