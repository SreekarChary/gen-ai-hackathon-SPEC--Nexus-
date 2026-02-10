"""Tests for Module 1 — Data Preprocessing."""
import pandas as pd
import pytest
from src.preprocessing import clean_data, engineer_features


def test_clean_data_handles_nans_and_junk():
    df = pd.DataFrame({
        "policy_number": ["POL1", "POL2"],
        "age": [30, 40],
        "_c39": [None, None],
        "incident_type": ["Collision", "?"],
        "target": ["Y", "N"]
    })
    result = clean_data(df)
    # _c39 should be dropped, '?' replaced and filled
    assert "_c39" not in result.columns
    assert result.isnull().sum().sum() == 0
    assert "?" not in result.values


def test_engineer_features_creates_correct_cols():
    df = pd.DataFrame({
        "policy_bind_date": ["2010-01-01"],
        "incident_date": ["2010-01-11"],
        "incident_hour_of_the_day": [12],
        "injury_claim": [1000],
        "property_claim": [1000],
        "total_claim_amount": [10000]
    })
    result = engineer_features(df)
    assert "days_to_incident" in result.columns
    assert result.iloc[0]["days_to_incident"] == 10
    assert "injury_claim_ratio" in result.columns
    assert "policy_bind_date" not in result.columns
