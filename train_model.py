import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

# Load dataset
data = pd.read_csv("synthetic_transactions.csv")

# Features
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

X = data[features]
y = data["is_fraud"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Create model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

# Train model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print("\n===== RiskGuard AI Model Results =====")

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)
print("\nConfusion Matrix:")
print(cm)

# Threshold tuning
print("\n===== Threshold Comparison =====")

thresholds = [0.50, 0.40, 0.30, 0.20]

for threshold in thresholds:

    y_threshold = (y_prob >= threshold).astype(int)

    precision_t = precision_score(
        y_test, y_threshold, zero_division=0
    )

    recall_t = recall_score(
        y_test, y_threshold, zero_division=0
    )

    f1_t = f1_score(
        y_test, y_threshold, zero_division=0
    )

    print(f"\nThreshold: {threshold}")
    print(f"Precision: {precision_t:.4f}")
    print(f"Recall   : {recall_t:.4f}")
    print(f"F1 Score : {f1_t:.4f}")
import joblib

joblib.dump(model, "risk_model.pkl")
print("\nModel saved successfully!")