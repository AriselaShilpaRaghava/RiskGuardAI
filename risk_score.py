import joblib
import pandas as pd

# Load trained model
model = joblib.load("risk_model.pkl")

# Example transaction
transaction = pd.DataFrame([{
    "transaction_amount": 25000,
    "transaction_hour": 2,
    "new_beneficiary": 1,
    "new_device": 1,
    "location_change": 1,
    "transaction_velocity": 6,
    "previous_risk": 0.75,
    "account_age_days": 100
}])

# Predict fraud probability
probability = model.predict_proba(transaction)[0][1]

# Convert probability to risk score
risk_score = probability * 100

# Determine risk level
if risk_score < 30:
    risk_level = "LOW"
elif risk_score < 70:
    risk_level = "MEDIUM"
else:
    risk_level = "HIGH"

print("\n===== RiskGuard AI Risk Assessment =====")
print(f"Fraud Probability : {probability:.4f}")
print(f"Risk Score        : {risk_score:.2f}/100")
print(f"Risk Level        : {risk_level}")