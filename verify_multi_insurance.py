import os
import sys

# Add root dir to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.predict import predict_claim

def test_vehicle():
    print("\n--- Testing Vehicle Insurance ---")
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
    result = predict_claim(sample, category="vehicle")
    print(f"Verdict: {result['verdict']}")
    print(f"Risk Score: {result['risk_score']}")
    print(f"Top Features: {[f['feature'] for f in result['top_features']]}")
    assert result['verdict'] in ['Fraudulent', 'Legitimate']

def test_health():
    print("\n--- Testing Health Insurance ---")
    sample = {
        "Patient_Age": "45",
        "Patient_Gender": "Female",
        "Claim_Amount": "5000",
        "Number_of_Procedures": "2",
        "Service_Date": "2023-01-01",
        "Claim_Date": "2023-01-05",
        "Policy_Expiration_Date": "2024-01-01",
        "Length_of_Stay_Days": "3",
        "Deductible_Amount": "500",
        "CoPay_Amount": "50",
        "Provider_Type": "Hospital",
        "Provider_Specialty": "Cardiology",
        "Claim_Submitted_Late": "False",
        "Number_of_Previous_Claims_Patient": "0"
    }
    result = predict_claim(sample, category="health")
    print(f"Verdict: {result['verdict']}")
    print(f"Risk Score: {result['risk_score']}")
    print(f"Top Features: {[f['feature'] for f in result['top_features']]}")
    assert result['verdict'] in ['Fraudulent', 'Legitimate']

if __name__ == "__main__":
    try:
        test_vehicle()
        test_health()
        print("\n✅ Verification script passed!")
    except Exception as e:
        print(f"\n❌ Verification failed: {e}")
        sys.exit(1)
