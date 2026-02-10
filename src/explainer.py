"""
Module 3 — Explainability
Uses SHAP to identify top contributing features and generates plain-English reasoning.
"""
import numpy as np
import shap

import config


def get_shap_values(model, X):
    """
    Compute SHAP values for a tree-based model.
    Returns the SHAP explainer and shap_values array.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # For binary classification, shap_values may be a list of 2 arrays;
    # we use the positive-class (index 1) values.
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    return explainer, shap_values


def top_features(shap_values_row, feature_names, n=None):
    """
    Given SHAP values for a single prediction, return the top-n features
    sorted by absolute contribution.
    Returns list of dicts: [{"feature": ..., "shap_value": ..., "direction": ...}, ...]
    """
    if n is None:
        n = config.TOP_N_FEATURES

    indices = np.argsort(np.abs(shap_values_row))[::-1][:n]
    results = []
    for i in indices:
        results.append({
            "feature": feature_names[i],
            "shap_value": round(float(shap_values_row[i]), 4),
            "direction": "increases fraud risk" if shap_values_row[i] > 0 else "decreases fraud risk",
        })
    return results


def generate_reasoning(top_feats: list, risk_score: float, verdict: str) -> str:
    """
    Produce a human-readable paragraph explaining why a claim was flagged.

    Parameters
    ----------
    top_feats : list of dicts from top_features()
    risk_score : float 0-1
    verdict : str "Fraudulent" or "Legitimate"
    """
    risk_pct = round(risk_score * 100, 1)

    if verdict == "Fraudulent":
        opening = (
            f"This claim carries a **high risk score of {risk_pct}%** and has been "
            f"classified as **Fraudulent**. The main red flags identified are:"
        )
    else:
        opening = (
            f"This claim has a **low risk score of {risk_pct}%** and has been "
            f"classified as **Legitimate**. Key factors supporting this assessment:"
        )

    bullet_points = []
    for i, feat in enumerate(top_feats, 1):
        name = feat["feature"].replace("_", " ").title()
        direction = feat["direction"]
        shap_val = feat["shap_value"]
        bullet_points.append(
            f"({i}) **{name}** — this factor {direction} "
            f"(impact score: {shap_val:+.4f})."
        )

    reasoning = opening + "\n\n" + "\n".join(bullet_points)

    if verdict == "Fraudulent":
        reasoning += (
            "\n\n⚠️ **Recommendation:** This claim should be escalated for "
            "manual investigation by a senior claims adjuster."
        )
    else:
        reasoning += (
            "\n\n✅ **Recommendation:** This claim appears legitimate and can "
            "proceed through normal processing."
        )

    return reasoning
