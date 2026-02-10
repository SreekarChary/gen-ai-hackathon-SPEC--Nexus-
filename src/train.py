import os
import sys
import argparse
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
)
from xgboost import XGBClassifier

# Add root dir to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def load_processed_data(category: str):
    """Load the preprocessed train/test split."""
    path = os.path.join(config.get_processed_dir(category), "processed_data.pkl")
    data = joblib.load(path)
    print(f"[✓] Loaded {category} data  —  {len(data['X_train'])} train / {len(data['X_test'])} test")
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

    print(f"\n{model_name} Results:")
    print(f"  Accuracy  : {acc:.4f}  |  ROC-AUC : {auc:.4f}")
    print(f"  Recall    : {rec:.4f}  |  Precision: {prec:.4f}")

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "roc_auc": auc}


def save_model(model, category: str, name: str):
    """Save a trained model to the models/<category>/ directory."""
    path_dir = config.get_model_dir(category)
    os.makedirs(path_dir, exist_ok=True)
    path = os.path.join(path_dir, f"{name}.pkl")
    joblib.dump(model, path)
    print(f"[✓] Model saved → {path}")


# ─── CLI entry point ───────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train insurance models.")
    parser.add_argument("--category", type=str, default=config.DEFAULT_CATEGORY, help="vehicle or health")
    args = parser.parse_args()
    
    cat = args.category
    data = load_processed_data(cat)
    X_train, X_test, y_train, y_test = data["X_train"], data["X_test"], data["y_train"], data["y_test"]

    # RF
    rf_model = train_random_forest(X_train, y_train)
    rf_metrics = evaluate(rf_model, X_test, y_test, "Random Forest")
    save_model(rf_model, cat, "random_forest")

    # XGB
    xgb_model = train_xgboost(X_train, y_train)
    xgb_metrics = evaluate(xgb_model, X_test, y_test, "XGBoost")
    save_model(xgb_model, cat, "xgboost")

    # Save Best
    model_dir = config.get_model_dir(cat)
    if rf_metrics["roc_auc"] >= xgb_metrics["roc_auc"]:
        print(f"🏆 RF wins for {cat}")
        save_model(rf_model, cat, "best_model")
        joblib.dump(rf_metrics, os.path.join(model_dir, "metrics.pkl"))
    else:
        print(f"🏆 XGB wins for {cat}")
        save_model(xgb_model, cat, "best_model")
        joblib.dump(xgb_metrics, os.path.join(model_dir, "metrics.pkl"))

    print(f"\n✅ Training complete for {cat}!")
