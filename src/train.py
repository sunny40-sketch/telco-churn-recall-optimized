import argparse
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, classification_report,
    precision_score, recall_score, f1_score
)


def build_pipeline(X_train: pd.DataFrame) -> Pipeline:
    cat_cols = X_train.select_dtypes(exclude="number").columns
    num_cols = X_train.select_dtypes(include="number").columns

    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ])

    model = LogisticRegression(max_iter=2000, class_weight="balanced")

    clf = Pipeline(steps=[
        ("prep", preprocessor),
        ("model", model),
    ])
    return clf


def pick_threshold_for_recall(y_true, y_proba, recall_target: float):
    thresholds = np.arange(0.05, 0.96, 0.01)

    best_th = None
    best_precision = -1.0
    best_stats = None

    for th in thresholds:
        y_hat = (y_proba >= th).astype(int)
        r = recall_score(y_true, y_hat, zero_division=0)
        if r >= recall_target:
            p = precision_score(y_true, y_hat, zero_division=0)
            f1 = f1_score(y_true, y_hat, zero_division=0)
            cm = confusion_matrix(y_true, y_hat)
            if p > best_precision:
                best_precision = p
                best_th = th
                best_stats = (p, r, f1, cm)

    if best_th is None:
        raise ValueError(
            f"No threshold achieved recall_target={recall_target}. "
            "Lower recall_target or check model/data."
        )

    return best_th, best_stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--recall_target", type=float, default=0.85)
    parser.add_argument("--test_size", type=float, default=0.25)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.data_path)

    # Target
    df["ChurnFlag"] = df["Churn"].map({"Yes": 1, "No": 0})

    # Important: convert TotalCharges to numeric (blanks -> NaN)
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    X = df.drop(columns=["customerID", "Churn", "ChurnFlag"])
    y = df["ChurnFlag"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y
    )

    clf = build_pipeline(X_train)
    clf.fit(X_train, y_train)

    # Baseline at default threshold (0.5 via predict)
    y_pred = clf.predict(X_test)
    print("=== Baseline (threshold=0.5 via predict) ===")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred, zero_division=0))
    print("Recall   :", recall_score(y_test, y_pred, zero_division=0))
    print("F1       :", f1_score(y_test, y_pred, zero_division=0))
    print("\nReport:\n", classification_report(y_test, y_pred, zero_division=0))

    # Recall-target threshold selection
    y_proba = clf.predict_proba(X_test)[:, 1]
    th, (p, r, f1, cm) = pick_threshold_for_recall(y_test, y_proba, args.recall_target)

    flagged = int(cm[0, 1] + cm[1, 1])
    fn = int(cm[1, 0])

    print("\n=== Recall-Target Operating Point ===")
    print("Recall target:", args.recall_target)
    print("Chosen threshold:", th)
    print("Precision:", p)
    print("Recall   :", r)
    print("F1       :", f1)
    print("Confusion Matrix:\n", cm)
    print("Flagged customers (TP+FP):", flagged)
    print("Missed churners (FN):", fn)


if __name__ == "__main__":
    main()
