"""Tests for Module 3 — Explainer."""
import numpy as np
import pytest
from src.explainer import top_features, generate_reasoning


def test_top_features_returns_correct_count():
    shap_values = np.array([0.3, -0.5, 0.1, 0.8, -0.2])
    names = ["feat_a", "feat_b", "feat_c", "feat_d", "feat_e"]
    result = top_features(shap_values, names, n=3)
    assert len(result) == 3


def test_top_features_sorted_by_abs_value():
    shap_values = np.array([0.3, -0.5, 0.1, 0.8, -0.2])
    names = ["feat_a", "feat_b", "feat_c", "feat_d", "feat_e"]
    result = top_features(shap_values, names, n=3)
    abs_vals = [abs(r["shap_value"]) for r in result]
    assert abs_vals == sorted(abs_vals, reverse=True)


def test_generate_reasoning_not_empty():
    feats = [
        {"feature": "claim_amount", "shap_value": 0.45, "direction": "increases fraud risk"},
        {"feature": "past_claims",  "shap_value": 0.30, "direction": "increases fraud risk"},
    ]
    text = generate_reasoning(feats, 0.87, "Fraudulent")
    assert len(text) > 50
    assert "87" in text
