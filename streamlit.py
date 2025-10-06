import streamlit as st
import pandas as pd
from joblib import load
from datetime import datetime
import shap
import matplotlib.pyplot as plt

# ---------------- Streamlit Setup ----------------
st.set_page_config(page_title="Fraud Detection", page_icon="💳", layout="centered")
st.title("💳 Fraud Detection App")
st.caption("Predict whether a transaction is fraudulent and understand which features influenced it.")
st.divider()

# ---------------- Load pipeline ----------------
pipeline = load("fraud_detection_pipeline.joblib")
scaler = pipeline.named_steps["scaler"]
model = pipeline.named_steps["model"]

# ---------------- User Inputs ----------------
amount = st.number_input("Transaction Amount ($)", min_value=0.0, step=0.01)
merchant_id = st.selectbox("Merchant ID", [109, 232, 394, 475, 688, 944])
transaction_type = st.selectbox("Transaction Type", ["purchase", "refund"])
location = st.selectbox(
    "Location",
    ["Dallas", "Houston", "Los Angeles", "New York", "Philadelphia",
     "Phoenix", "San Antonio", "San Diego", "San Jose"]
)

date = st.date_input("Transaction Date", datetime.now().date())
time = st.time_input("Transaction Time", datetime.now().time())
dt = datetime.combine(date, time)
hour, day, month = dt.hour, dt.day, dt.month

# Build input dataframe
X_input = pd.DataFrame([{
    "Amount": amount,
    "MerchantID": merchant_id,
    "Hour": hour,
    "Day": day,
    "Month": month,
    "TransactionType_refund": 1 if transaction_type == "refund" else 0,
    "Location_Dallas": 1 if location == "Dallas" else 0,
    "Location_Houston": 1 if location == "Houston" else 0,
    "Location_Los Angeles": 1 if location == "Los Angeles" else 0,
    "Location_New York": 1 if location == "New York" else 0,
    "Location_Philadelphia": 1 if location == "Philadelphia" else 0,
    "Location_Phoenix": 1 if location == "Phoenix" else 0,
    "Location_San Antonio": 1 if location == "San Antonio" else 0,
    "Location_San Diego": 1 if location == "San Diego" else 0,
    "Location_San Jose": 1 if location == "San Jose" else 0
}])

st.divider()

# ---------------- Predict & Explain ----------------
if st.button("🔍 Predict Fraud"):
    # --- Prediction ---
    pred = pipeline.predict(X_input)[0]
    proba = pipeline.predict_proba(X_input)[0][1]

    if pred == 1:
        st.error(f"🚨 Fraudulent Transaction Detected! (Probability: {proba:.2f})")
    else:
        st.success(f"✅ Legitimate Transaction. (Fraud Probability: {proba:.2f})")

    st.divider()
    st.subheader("🧠 Model Explainability")

    # --- Prepare transformed input for SHAP ---
    X_transformed = scaler.transform(X_input)

    # --- SHAP on model directly ---
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_transformed)

    # For binary classification, take class 1 (fraud)
    if shap_values.values.ndim == 3:
        shap_single = shap.Explanation(
            values=shap_values.values[:, :, 1],
            base_values=shap_values.base_values[:, 1],
            data=X_input.values,
            feature_names=X_input.columns
        )
    else:
        shap_single = shap_values

    # ---- Waterfall plot ----
    st.write("### 🔍 Feature impact for this prediction")
    fig = plt.figure(figsize=(8, 6))
    shap.plots.waterfall(shap_single[0], max_display=15, show=False)
    st.pyplot(fig, clear_figure=True)

    # ---- SHAP Value Table ----
    st.write("### 📊 Feature contributions")
    shap_df = pd.DataFrame({
        "Feature": X_input.columns,
        "SHAP Value": shap_single.values[0]
    }).sort_values(by="SHAP Value", key=abs, ascending=False)
    st.dataframe(shap_df, use_container_width=True)

    # ---- Optional bar chart ----
    st.bar_chart(shap_df.set_index("Feature")["SHAP Value"])
