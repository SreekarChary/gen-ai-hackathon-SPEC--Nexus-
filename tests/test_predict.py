"""Tests for the Prediction / Inference module."""
import os
import pytest
import config


MODEL_EXISTS = os.path.exists(os.path.join(config.MODELS_DIR, "random_forest.pkl"))


@pytest.mark.skipif(not MODEL_EXISTS, reason="Model not yet trained")
def test_predict_claim_returns_dict():
    from src.predict import predict_claim
    # Must match the fields expected by predict.py and web form
    sample = {
        "incident_severity": "Major Damage",
        "incident_type": "Single Vehicle Collision",
        "collision_type": "Side Collision",
        "authorities_contacted": "Police",
        "incident_hour_of_the_day": "5",
        "number_of_vehicles_involved": "1",
        "bodily_injuries": "1",
        "witnesses": "2",
        "police_report_available": "YES",
        "total_claim_amount": "50000",
        "injury_claim": "5000",
        "property_claim": "5000",
        "vehicle_claim": "40000",
        "policy_annual_premium": "1500",
        "policy_deductable": "1000",
        "umbrella_limit": "0",
        "age": "35",
        "insured_sex": "MALE",
        "insured_education_level": "PhD",
        "incident_state": "OH",
        "auto_make": "Saab",
        "policy_bind_date": "2010-01-01",
        "incident_date": "2015-01-01"
    }
    result = predict_claim(sample)
    assert "verdict" in result
    assert "risk_score" in result
    assert "reasoning" in result
    assert result["verdict"] in ("Fraudulent", "Legitimate")
    assert 0.0 <= result["risk_score"] <= 1.0
