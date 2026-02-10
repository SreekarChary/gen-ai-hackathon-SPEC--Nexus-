"""
Module 2 — Model Training
Trains Random Forest and XGBoost classifiers, evaluates them, and saves the best.
"""
import os
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
)
from xgboost import XGBClassifier

import config


def load_processed_data():
    """Load the preprocessed train/test split."""
    data = joblib.load(os.path.join(config.DATA_PROCESSED_DIR, "processed_data.pkl"))
    print(f"[✓] Loaded processed data  —  {len(data['X_train'])} train / {len(data['X_test'])} test")
    return data


def train_random_forest(X_train, y_train):
    """Train a Random Forest classifier."""
    model = RandomForestClassifier(**config.RANDOM_FOREST_PARAMS)
    model.fit(X_train, y_train)
    print("[✓] Random Forest trained")
    return model


def train_xgboost(X_train, y_train):
    """Train an XGBoost classifier."""
    model = XGBClassifier(**config.XGBOOST_PARAMS)
    model.fit(X_train, y_train)
    print("[✓] XGBoost trained")
    return model


def evaluate(model, X_test, y_test, model_name="Model"):
    """Print comprehensive evaluation metrics."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n{'='*50}")
    print(f"  {model_name} — Evaluation Metrics")
    print(f"{'='*50}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"  ROC-AUC   : {auc:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"    {cm}")
    print(f"\n{classification_report(y_test, y_pred)}")

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "roc_auc": auc}


def save_model(model, name: str):
    """Save a trained model to the models/ directory."""
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    path = os.path.join(config.MODELS_DIR, f"{name}.pkl")
    joblib.dump(model, path)
    print(f"[✓] Model saved → {path}")


# ─── CLI entry point ───────────────────────────────────
if __name__ == "__main__":
    data = load_processed_data()
    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]

    # Train Random Forest
    rf_model = train_random_forest(X_train, y_train)
    rf_metrics = evaluate(rf_model, X_test, y_test, "Random Forest")
    save_model(rf_model, "random_forest")

    # Train XGBoost
    xgb_model = train_xgboost(X_train, y_train)
    xgb_metrics = evaluate(xgb_model, X_test, y_test, "XGBoost")
    save_model(xgb_model, "xgboost")

    # Summary
    print("\n" + "=" * 50)
    print("  Best Model Summary")
    print("=" * 50)
    if rf_metrics["roc_auc"] >= xgb_metrics["roc_auc"]:
        print("  🏆 Random Forest wins with ROC-AUC:", f"{rf_metrics['roc_auc']:.4f}")
    else:
        print("  🏆 XGBoost wins with ROC-AUC:", f"{xgb_metrics['roc_auc']:.4f}")

    print("\n✅ Model training complete!")
