"""Tests for the Prediction / Inference module."""
# These tests require a trained model.
# They will be skipped if model files are not found.
import os
import pytest
import config


MODEL_EXISTS = os.path.exists(os.path.join(config.MODELS_DIR, "random_forest.pkl"))


@pytest.mark.skipif(not MODEL_EXISTS, reason="Model not yet trained")
def test_predict_claim_returns_dict():
    from src.predict import predict_claim
    sample = {
        "policy_number": "POL-99999",
        "claim_amount": "20000",
        "incident_type": "collision",
        "incident_severity": "major",
        "insured_age": "40",
        "annual_income": "60000",
        "number_of_past_claims": "3",
        "police_report_filed": "No",
    }
    result = predict_claim(sample)
    assert "verdict" in result
    assert "risk_score" in result
    assert "reasoning" in result
    assert result["verdict"] in ("Fraudulent", "Legitimate")
    assert 0.0 <= result["risk_score"] <= 1.0
