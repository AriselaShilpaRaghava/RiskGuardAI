import joblib
import pandas as pd
import numpy as np

# Load model
model = joblib.load("risk_model.pkl")

# Load test dataset
data = pd.read_csv("synthetic_transactions.csv")

features_to_test = [
    "new_beneficiary",
    "new_device",
    "location_change",
    "transaction_velocity",
    "previous_risk"
]

# Use 100 transactions
test_data = data.sample(100, random_state=42)

total_tests = 0
consistent_tests = 0

print("\n===== RiskGuard AI Multi-Transaction Stability Test =====")

for _, row in test_data.iterrows():

    original = pd.DataFrame([{
        "transaction_amount": row["transaction_amount"],
        "transaction_hour": row["transaction_hour"],
        "new_beneficiary": row["new_beneficiary"],
        "new_device": row["new_device"],
        "location_change": row["location_change"],
        "transaction_velocity": row["transaction_velocity"],
        "previous_risk": row["previous_risk"],
        "account_age_days": row["account_age_days"]
    }])

    original_probability = model.predict_proba(original)[0][1]

    for feature in features_to_test:

        modified = original.copy()

        # Change binary features
        if feature in [
            "new_beneficiary",
            "new_device",
            "location_change"
        ]:
            modified[feature] = 1 - modified[feature].iloc[0]

        # Reduce continuous risk features
        elif feature == "transaction_velocity":
            modified[feature] = 1

        elif feature == "previous_risk":
            modified[feature] = 0.10

        new_probability = model.predict_proba(modified)[0][1]

        # Determine whether risk moved in expected direction
        original_value = original[feature].iloc[0]
        new_value = modified[feature].iloc[0]

        risk_increased = new_probability > original_probability
        feature_increased = new_value > original_value

        # For risk-related features:
        # increasing feature should generally increase risk
        expected_direction = feature_increased

        if risk_increased == expected_direction:
            consistent_tests += 1

        total_tests += 1

stability_score = (
    consistent_tests / total_tests
) * 100

print(f"\nTotal Tests        : {total_tests}")
print(f"Consistent Tests   : {consistent_tests}")
print(f"Inconsistent Tests : {total_tests - consistent_tests}")
print(f"\nExplanation Stability Score: {stability_score:.2f}%")