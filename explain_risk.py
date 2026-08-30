import joblib
import pandas as pd
import shap

# Load trained model
model = joblib.load("risk_model.pkl")

# Feature names
features = [
    "transaction_amount",
    "transaction_hour",
    "new_beneficiary",
    "new_device",
    "location_change",
    "transaction_velocity",
    "previous_risk",
    "account_age_days"
]

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

# Create SHAP explainer
explainer = shap.TreeExplainer(model)

# Calculate SHAP values
shap_values = explainer.shap_values(transaction)

# Handle SHAP output
if isinstance(shap_values, list):
    values = shap_values[1][0]
else:
    values = shap_values[0, :, 1] if shap_values.ndim == 3 else shap_values[0]

# Display feature contributions
print("\n===== RiskGuard AI Explanation =====")

for feature, value in zip(features, values):
    print(f"{feature:25} {value:+.4f}")