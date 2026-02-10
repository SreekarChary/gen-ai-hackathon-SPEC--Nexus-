"""
Prediction / Inference Module
Loads trained model + encoders, runs prediction on new claim data.
"""
import os
import numpy as np
import pandas as pd
import joblib

import config
from src.explainer import get_shap_values, top_features, generate_reasoning


def load_artifacts(model_name="random_forest"):
    """Load model, encoders, and scaler from models/ directory."""
    model = joblib.load(os.path.join(config.MODELS_DIR, f"{model_name}.pkl"))
    label_encoders = joblib.load(os.path.join(config.MODELS_DIR, "label_encoders.pkl"))
    scaler = joblib.load(os.path.join(config.MODELS_DIR, "scaler.pkl"))
    data = joblib.load(os.path.join(config.DATA_PROCESSED_DIR, "processed_data.pkl"))
    feature_names = data["feature_names"]
    return model, label_encoders, scaler, feature_names


def preprocess_input(claim_dict: dict, label_encoders: dict, scaler, feature_names: list):
    """
    Transform a raw claim dictionary into a model-ready numpy array.
    Applies the same label encoding and scaling used during training.
    """
    df = pd.DataFrame([claim_dict])

    # Label-encode categoricals
    for col, le in label_encoders.items():
        if col in df.columns:
            # Handle unseen labels gracefully
            known = set(le.classes_)
            df[col] = df[col].apply(lambda x: x if x in known else le.classes_[0])
            df[col] = le.transform(df[col].astype(str))

    # Ensure correct column order and fill missing with 0
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_names]

    # Scale numeric columns
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df[num_cols] = scaler.transform(df[num_cols])

    return df


def predict_claim(claim_dict: dict, model_name="random_forest"):
    """
    Full prediction pipeline for a single claim.
    Returns dict: {verdict, risk_score, reasoning, top_features}
    """
    model, label_encoders, scaler, feature_names = load_artifacts(model_name)
    X = preprocess_input(claim_dict, label_encoders, scaler, feature_names)

    # Probability of fraud (class 1)
    risk_score = float(model.predict_proba(X)[0][1])
    verdict = "Fraudulent" if risk_score >= 0.5 else "Legitimate"

    # Explainability
    _, shap_vals = get_shap_values(model, X)
    shap_row = shap_vals[0] if shap_vals.ndim > 1 else shap_vals
    top_feats = top_features(shap_row, feature_names)
    reasoning = generate_reasoning(top_feats, risk_score, verdict)

    return {
        "verdict": verdict,
        "risk_score": round(risk_score, 4),
        "reasoning": reasoning,
        "top_features": top_feats,
    }
