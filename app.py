import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Page setup
st.set_page_config(
    page_title="AI Fraud Risk Scoring System",
    page_icon="🚨",
    layout="wide"
)

# Title
st.title("🚨 AI Financial Fraud Detection & Risk Scoring System")

st.write(
    "This system uses machine learning and audit-oriented risk logic to estimate fraud risk in financial transactions."
)

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("data/creditcard.csv")

data = load_data()

# Prepare data
X = data.drop("Class", axis=1)
y = data["Class"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Train model
@st.cache_resource
def train_model():
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight={0: 1, 1: 10}
    )
    model.fit(X_train, y_train)
    return model

model = train_model()

# Risk level function
def risk_level(prob):
    if prob >= 0.20:
        return "🔴 High Risk"
    elif prob >= 0.05:
        return "🟡 Medium Risk"
    else:
        return "🟢 Low Risk"

# Sidebar inputs
st.sidebar.header("Enter Transaction Details")

time = st.sidebar.number_input("Time", value=10000)
amount = st.sidebar.number_input("Amount", value=100.0)

# Create input sample
input_data = X.iloc[0:1].copy()
input_data["Time"] = time
input_data["Amount"] = amount

input_scaled = scaler.transform(input_data)

# Prediction button
if st.sidebar.button("Analyze Transaction"):

    model_prob = model.predict_proba(input_scaled)[0][1]

    # Rule-based risk adjustment for audit-style interpretation
    adjusted_prob = model_prob

    risk_factors = []

    if amount > 2000:
        adjusted_prob += 0.10
        risk_factors.append("High transaction amount")

    if amount > 5000:
        adjusted_prob += 0.15
        risk_factors.append("Very high transaction amount")

    if time < 1000 or time > 150000:
        adjusted_prob += 0.05
        risk_factors.append("Unusual transaction timing")

    adjusted_prob = min(adjusted_prob, 1.0)

    risk = risk_level(adjusted_prob)

    st.subheader("Result")

    col1, col2, col3 = st.columns(3)

    col1.metric("Model Fraud Probability", f"{model_prob:.4f}")
    col2.metric("Adjusted Risk Score", f"{adjusted_prob:.4f}")
    col3.metric("Risk Level", risk)

    st.markdown("---")

    st.subheader("Audit-Oriented Interpretation")

    if risk == "🔴 High Risk":
        st.error("This transaction should be reviewed immediately by a fraud or audit team.")
    elif risk == "🟡 Medium Risk":
        st.warning("This transaction may require additional monitoring or manual review.")
    else:
        st.success("This transaction is currently classified as low risk.")

    if risk_factors:
        st.write("Detected risk factors:")
        for factor in risk_factors:
            st.write(f"- {factor}")
    else:
        st.write("No major rule-based risk factors detected.")

    st.markdown("---")

    st.subheader("Transaction Input Summary")
    st.write(f"Time: {time}")
    st.write(f"Amount: ${amount:,.2f}")

else:
    st.info("Enter transaction details in the sidebar and click **Analyze Transaction**.")