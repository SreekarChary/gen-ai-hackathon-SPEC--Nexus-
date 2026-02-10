import os
import sys

# Add root dir to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.predict import predict_claim

def test_health_fraud():
    print("\n--- Testing Health Insurance (Fraud Sample) ---")
    # Row from CSV: 85973291,XAI215993963,8762,2025-01-19,2025-01-17,2029-04-07,1883481.3,40,Other,Washington,CO,408,Laboratory,Cardiology,Dallas,AZ,J02.9,99203,2,Elective,Deceased,0,Outpatient,3618.4,851.43,0,6,170.4,False,True
    sample = {
        "Patient_Age": "40",
        "Patient_Gender": "Other",
        "Claim_Amount": "1883481.3",
        "Number_of_Procedures": "2",
        "Service_Date": "2025-01-17",
        "Claim_Date": "2025-01-19",
        "Policy_Expiration_Date": "2029-04-07",
        "Length_of_Stay_Days": "0",
        "Deductible_Amount": "3618.4",
        "CoPay_Amount": "851.43",
        "Provider_Type": "Laboratory",
        "Provider_Specialty": "Cardiology",
        "Claim_Submitted_Late": "False",
        "Number_of_Previous_Claims_Patient": "0"
    }
    result = predict_claim(sample, category="health")
    print(f"Verdict: {result['verdict']}")
    print(f"Risk Score: {result['risk_score']}")
    print(f"Top Features: {[f['feature'] for f in result['top_features']]}")
    
if __name__ == "__main__":
    test_health_fraud()
