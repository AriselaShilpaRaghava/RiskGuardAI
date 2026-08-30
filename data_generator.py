import pandas as pd
import numpy as np

np.random.seed(42)

n = 10000

# Basic transaction information
data = pd.DataFrame({
    "transaction_id": range(1, n + 1),
    "transaction_amount": np.random.lognormal(
        mean=7, sigma=1, size=n
    ),
    "transaction_hour": np.random.randint(0, 24, n),
    "new_beneficiary": np.random.binomial(1, 0.15, n),
    "new_device": np.random.binomial(1, 0.10, n),
    "location_change": np.random.binomial(1, 0.08, n),
    "transaction_velocity": np.random.poisson(2, n),
    "previous_risk": np.random.uniform(0, 1, n),
    "account_age_days": np.random.randint(1, 2000, n)
})

# Start with low baseline fraud probability
fraud_probability = np.full(n, 0.03)

# Add risk when suspicious conditions exist
fraud_probability += np.where(
    data["new_beneficiary"] == 1, 0.12, 0
)

fraud_probability += np.where(
    data["new_device"] == 1, 0.15, 0
)

fraud_probability += np.where(
    data["location_change"] == 1, 0.12, 0
)

fraud_probability += np.where(
    data["transaction_velocity"] >= 5, 0.20, 0
)

fraud_probability += np.where(
    data["previous_risk"] >= 0.70, 0.20, 0
)

# Unusual transaction time
night_transaction = (
    (data["transaction_hour"] >= 0) &
    (data["transaction_hour"] <= 5)
)

fraud_probability += np.where(
    night_transaction, 0.10, 0
)

# Large transaction
fraud_probability += np.where(
    data["transaction_amount"] >= 10000, 0.12, 0
)

# Cap probability between 0 and 0.95
fraud_probability = np.clip(
    fraud_probability, 0, 0.95
)

# Generate fraud label
data["is_fraud"] = np.random.binomial(
    1, fraud_probability
)

# Save dataset
data.to_csv(
    "synthetic_transactions.csv",
    index=False
)

print("New dataset created successfully!")
print("\nDataset shape:", data.shape)

print("\nFraud distribution:")
print(data["is_fraud"].value_counts())

print("\nFraud percentage:")
print(
    data["is_fraud"].mean() * 100,
    "%"
)

print("\nFirst 5 rows:")
print(data.head())