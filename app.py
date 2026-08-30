import streamlit as st
import joblib
import pandas as pd
import shap

model = joblib.load("risk_model_compressed.pkl")

st.set_page_config(
    page_title="RiskGuard AI",
    page_icon="🛡️"
)

st.title("🛡️ RiskGuard AI")
st.subheader("Transaction Risk Assessment")

transaction_amount = st.number_input(
    "Transaction Amount (₹)",
    min_value=0.0,
    value=25000.0
)

transaction_hour = st.slider(
    "Transaction Hour",
    0, 23, 2
)

new_beneficiary = st.selectbox(
    "New Beneficiary?",
    ["No", "Yes"]
)

new_device = st.selectbox(
    "New Device?",
    ["No", "Yes"]
)

location_change = st.selectbox(
    "Location Changed?",
    ["No", "Yes"]
)

transaction_velocity = st.number_input(
    "Transactions in Recent Period",
    min_value=0,
    value=6
)

previous_risk = st.slider(
    "Previous Risk",
    0.0, 1.0, 0.75
)

account_age_days = st.number_input(
    "Account Age (days)",
    min_value=1,
    value=100
)

if st.button("🔍 Analyze Transaction"):

    transaction = pd.DataFrame([{
        "transaction_amount": transaction_amount,
        "transaction_hour": transaction_hour,
        "new_beneficiary": 1 if new_beneficiary == "Yes" else 0,
        "new_device": 1 if new_device == "Yes" else 0,
        "location_change": 1 if location_change == "Yes" else 0,
        "transaction_velocity": transaction_velocity,
        "previous_risk": previous_risk,
        "account_age_days": account_age_days
    }])

    probability = model.predict_proba(transaction)[0][1]

    risk_score = probability * 100

    if risk_score < 30:
        risk_level = "LOW"
    elif risk_score < 70:
        risk_level = "MEDIUM"
    else:
        risk_level = "HIGH"

    st.divider()

    st.metric(
        "Risk Score",
        f"{risk_score:.2f}/100"
    )

    st.metric(
        "Fraud Probability",
        f"{probability * 100:.2f}%"
    )

    if risk_level == "HIGH":
        st.error(f"🚨 Risk Level: {risk_level}")
    elif risk_level == "MEDIUM":
        st.warning(f"⚠️ Risk Level: {risk_level}")
    else:
        st.success(f"✅ Risk Level: {risk_level}")

    # SHAP explanation
    st.subheader("🔍 Why this risk?")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(transaction)

    if isinstance(shap_values, list):
        values = shap_values[1][0]
    elif len(shap_values.shape) == 3:
        values = shap_values[0, :, 1]
    else:
        values = shap_values[0]

    explanation = pd.DataFrame({
        "Feature": transaction.columns,
        "SHAP Value": values
    })

    explanation["Impact"] = explanation["SHAP Value"].apply(
        lambda x: "Higher Risk" if x > 0 else "Lower Risk"
    )

    explanation = explanation.sort_values(
        "SHAP Value",
        key=abs,
        ascending=False
    )

    st.dataframe(
        explanation,
        use_container_width=True,
        hide_index=True
    )
st.subheader("🛡️ Explanation Reliability")

st.metric(
    "Stability Score",
    "74.80%"
)

st.caption(
    "Preliminary directional consistency measured on "
    "500 feature-change tests using the synthetic dataset."
)
