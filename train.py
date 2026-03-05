import argparse
import json
import os
import numpy as np
import pandas as pd
import joblib

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
    cat_cols = X_train.select_dtypes(exclude="number").columns.tolist()
    num_cols = X_train.select_dtypes(include="number").columns.tolist()

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

            # Pick the threshold that meets recall target with highest precision
            if p > best_precision:
                best_precision = p
                best_th = float(th)
                best_stats = (float(p), float(r), float(f1), cm)

    if best_th is None:
        raise ValueError(
            f"No threshold achieved recall_target={recall_target}. "
            "Lower recall_target or improve model/features."
        )

    return best_th, best_stats


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def main(args=None):
    parser = argparse.ArgumentParser(description="Train Telco churn pipeline + choose recall-optimized threshold.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to WA_Fn-UseC_-Telco-Customer-Churn.csv")
    parser.add_argument("--recall_target", type=float, default=0.85, help="Minimum recall required for class=1 (churn)")
    parser.add_argument("--test_size", type=float, default=0.25)
    parser.add_argument("--random_state", type=int, default=42)

    # NEW: outputs
    parser.add_argument("--out_dir", type=str, default="artifacts", help="Where to save model + config")
    parser.add_argument("--model_name", type=str, default="telco_churn_pipeline.pkl")
    parser.add_argument("--config_name", type=str, default="telco_threshold_config.json")
    args = parser.parse_args(args)

    ensure_dir(args.out_dir)

    df = pd.read_csv(args.data_path)

    # Basic sanity checks
    required = {"Churn"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Target
    df["ChurnFlag"] = df["Churn"].map({"Yes": 1, "No": 0})
    if df["ChurnFlag"].isna().any():
        bad = df.loc[df["ChurnFlag"].isna(), "Churn"].unique()
        raise ValueError(f"Unexpected values in Churn column: {bad}")

    # Convert TotalCharges if present (blanks -> NaN)
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Drop known non-features if present
    drop_cols = [c for c in ["customerID", "Churn", "ChurnFlag"] if c in df.columns]
    X = df.drop(columns=drop_cols)
    y = df["ChurnFlag"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y
    )

    clf = build_pipeline(X_train)
    clf.fit(X_train, y_train)

    # Baseline @ 0.5 (predict uses default threshold)
    y_pred = clf.predict(X_test)
    base_cm = confusion_matrix(y_test, y_pred)
    base_p = precision_score(y_test, y_pred, zero_division=0)
    base_r = recall_score(y_test, y_pred, zero_division=0)
    base_f1 = f1_score(y_test, y_pred, zero_division=0)

    print("\n=== Baseline (threshold=0.5 via predict) ===")
    print("Confusion Matrix:\n", base_cm)
    print(f"Precision: {base_p:.4f}")
    print(f"Recall   : {base_r:.4f}")
    print(f"F1       : {base_f1:.4f}")
    print("\nReport:\n", classification_report(y_test, y_pred, zero_division=0))

    # Recall-target threshold selection
    y_proba = clf.predict_proba(X_test)[:, 1]
    th, (p, r, f1, cm) = pick_threshold_for_recall(y_test, y_proba, args.recall_target)

    flagged = int(cm[0, 1] + cm[1, 1])  # FP + TP
    fn = int(cm[1, 0])                  # missed churners

    print("\n=== Recall-Target Operating Point ===")
    print("Recall target:", args.recall_target)
    print("Chosen threshold:", th)
    print(f"Precision: {p:.4f}")
    print(f"Recall   : {r:.4f}")
    print(f"F1       : {f1:.4f}")
    print("Confusion Matrix:\n", cm)
    print("Flagged customers (TP+FP):", flagged)
    print("Missed churners (FN):", fn)

    # NEW: Save ONE pipeline (prep + model)
    model_path = os.path.join(args.out_dir, args.model_name)
    joblib.dump(clf, model_path)

    # NEW: Save threshold config
    config = {
        "recall_target": float(args.recall_target),
        "chosen_threshold": float(th),
        "test_size": float(args.test_size),
        "random_state": int(args.random_state),
        "baseline": {
            "precision": float(base_p),
            "recall": float(base_r),
            "f1": float(base_f1),
            "confusion_matrix": base_cm.tolist(),
        },
        "operating_point": {
            "precision": float(p),
            "recall": float(r),
            "f1": float(f1),
            "confusion_matrix": cm.tolist(),
            "flagged_customers_tp_fp": int(flagged),
            "missed_churners_fn": int(fn),
        },
        "feature_columns_expected": list(X.columns),
    }

    config_path = os.path.join(args.out_dir, args.config_name)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print("\n=== Saved Artifacts ===")
    print("Pipeline:", model_path)
    print("Config  :", config_path)
    print("Expected input columns (for Streamlit):", list(X.columns))


if __name__ == "__main__":
    # Example: provide a dummy path for local testing in Colab
    # In a real scenario, this would be a valid path to your data file.
    main(args=['--data_path', r"C:\Users\kandu\OneDrive\Pictures\Desktop\Chrun_project\WA_Fn-UseC_-Telco-Customer-Churn-checkpoint.csv"])