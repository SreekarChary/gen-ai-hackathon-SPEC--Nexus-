import os
import sys
import joblib
import pandas as pd
import numpy as np

# Add root dir to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from src.predict import load_artifacts, preprocess_input

def debug_prediction():
    category = "vehicle"
    model, label_encoders, scaler, feature_names = load_artifacts(category)
    
    sample = {
        "incident_severity": "Major Damage",
        "incident_type": "Single Vehicle Collision",
        "incident_hour_of_the_day": "5",
        "number_of_vehicles_involved": "1",
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
    
    X = preprocess_input(sample, category, label_encoders, scaler, feature_names)
    
    print("\nPreprocessed columns:")
    print(X.columns.tolist())
    
    if hasattr(model, "feature_names_in_"):
        print("\nModel expected features (feature_names_in_):")
        print(model.feature_names_in_.tolist())
        
        # Check diff
        unseen = set(X.columns) - set(model.feature_names_in_)
        missing = set(model.feature_names_in_) - set(X.columns)
        print(f"\nUnseen in model: {unseen}")
        print(f"Missing in input: {missing}")
    else:
        print("\nModel has no feature_names_in_ attribute.")

if __name__ == "__main__":
    debug_prediction()
