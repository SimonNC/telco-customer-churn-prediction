# app/streamlit_app.py

from pathlib import Path
import joblib
import pandas as pd
import streamlit as st


# -----------------------------
# App config
# -----------------------------
st.set_page_config(
    page_title="Telco Churn Predictor",
    page_icon="📉",
    layout="centered",
)

st.title("📉 Telco Customer Churn Predictor")
st.write(
    "Predict churn risk for a single customer based on contract, tenure, pricing, and service features.\n"
    "This app uses the **trained Random Forest pipeline** (preprocessing + model)."
)

# -----------------------------
# Load model artifact
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "churn_model_rf.joblib"

@st.cache_resource
def load_model(path: Path):
    return joblib.load(path)

if not MODEL_PATH.exists():
    st.error(f"Model file not found at: {MODEL_PATH}")
    st.stop()

model = load_model(MODEL_PATH)


# -----------------------------
# Inputs (match training features)
# -----------------------------
st.subheader("Customer inputs")

col1, col2 = st.columns(2)

with col1:
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    internet_service = st.selectbox("InternetService", ["DSL", "Fiber optic", "No"])
    paperless_billing = st.selectbox("PaperlessBilling", ["Yes", "No"])
    payment_method = st.selectbox(
        "PaymentMethod",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
    )

with col2:
    tenure = st.number_input("tenure (months)", min_value=0, max_value=72, value=12, step=1)
    monthly_charges = st.number_input("MonthlyCharges", min_value=0.0, max_value=200.0, value=70.0, step=1.0)

    # TotalCharges can be provided; default suggestion uses a simple proxy (tenure * monthly)
    default_total = float(tenure) * float(monthly_charges)
    total_charges = st.number_input(
        "TotalCharges",
        min_value=0.0,
        max_value=10000.0,
        value=round(default_total, 2),
        step=10.0,
        help="If unknown, a rough proxy is tenure × MonthlyCharges.",
    )

# TechSupport / OnlineSecurity depend on InternetService in the original dataset
st.caption("Note: TechSupport and OnlineSecurity are typically applicable only for Internet customers.")

if internet_service == "No":
    tech_support = "No internet service"
    online_security = "No internet service"
    st.info("InternetService = No → TechSupport and OnlineSecurity set to 'No internet service'.")
else:
    tech_support = st.selectbox("TechSupport", ["Yes", "No"])
    online_security = st.selectbox("OnlineSecurity", ["Yes", "No"])


# -----------------------------
# Prediction controls
# -----------------------------
st.subheader("Prediction")
threshold = st.slider(
    "Decision threshold (classify as churn if probability ≥ threshold)",
    min_value=0.1,
    max_value=0.9,
    value=0.5,
    step=0.05,
)
st.caption(
    "Lower thresholds increase churn detection (higher recall) "
    "but may trigger more false positives."
)


predict_btn = st.button("Predict churn risk")


# -----------------------------
# Run prediction
# -----------------------------
if predict_btn:
    # Build a single-row dataframe with EXACT feature names expected by the pipeline
    input_df = pd.DataFrame([{
        "Contract": contract,
        "TechSupport": tech_support,
        "OnlineSecurity": online_security,
        "InternetService": internet_service,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }])

    proba = float(model.predict_proba(input_df)[:, 1][0])
    pred = int(proba >= threshold)

    st.metric("Churn probability", f"{proba:.2%}")

    if pred == 1:
        st.error("High churn risk → prioritize retention action.")
        st.write(
            "- Consider offering a contract upgrade incentive (month-to-month → 1-year/2-year)\n"
            "- Proactive support / onboarding check\n"
            "- Bundle protective services (TechSupport / OnlineSecurity) if applicable"
        )
    else:
        st.success("Lower churn risk → standard monitoring.")
        st.write(
            "- Continue regular customer success touchpoints\n"
            "- Monitor risk changes (billing issues, service changes, tenure stage)"
        )

    st.markdown("---")
    st.caption("Model: Random Forest pipeline (preprocessing + classifier) exported via joblib.")

st.markdown("---")

st.markdown(
    """
    <div style="text-align: center; font-size: 0.9em; color: gray;">
        📂 <a href="https://github.com/SimonNC/telco-customer-churn-prediction" target="_blank">
        View source code on GitHub</a>
    </div>
    """,
    unsafe_allow_html=True
)




