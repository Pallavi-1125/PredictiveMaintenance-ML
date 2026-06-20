import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------
# Login Credentials
# -------------------------------
USERNAME = "admin"
PASSWORD = "predict123"

# Session state
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False


# -------------------------------
# Login Function
# -------------------------------
def login():
    st.title("🔐 Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == USERNAME and password == PASSWORD:
            st.session_state.logged_in = True
            st.success("Login successful ✅")

            # For older Streamlit versions
            st.experimental_rerun()

        else:
            st.error("Invalid username or password")


# Show login page first
if not st.session_state.logged_in:
    login()
    st.stop()

# -------------------------------
# Load Model
# -------------------------------
model = joblib.load("models/predictive_maintenance_model.pkl")

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.title("📌 Project Information")

st.sidebar.info("""
Predictive Maintenance using Machine Learning

Model Used:
Random Forest Classifier

Accuracy:
92.4%

Purpose:
Predict machine failure before breakdown.
""")

# Logout button
if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.experimental_rerun()


# -------------------------------
# Main Title
# -------------------------------
st.title("🔧 Predictive Maintenance System")

st.markdown("""
## About this Project

This application predicts whether an industrial machine is likely to fail based on sensor data.

Input Parameters:
- Machine Type
- Air Temperature
- Process Temperature
- Rotational Speed
- Torque
- Tool Wear

Output:
- Machine Healthy
- Machine Failure Predicted
""")

# -------------------------------
# Inputs
# -------------------------------
machine_type = st.selectbox(
    "Machine Type",
    ["H", "L", "M"]
)

# Encoding mapping
# H -> 0, L -> 1, M -> 2
mapping = {
    "H": 0,
    "L": 1,
    "M": 2
}

air_temp = st.number_input("Air Temperature [K]", value=300.0)
process_temp = st.number_input("Process Temperature [K]", value=310.0)
rpm = st.number_input("Rotational Speed [rpm]", value=1500.0)
torque = st.number_input("Torque [Nm]", value=40.0)
tool_wear = st.number_input("Tool Wear [min]", value=120.0)

# -------------------------------
# Input Visualization
# -------------------------------
st.subheader("📊 Input Summary")

chart_data = pd.DataFrame({
    "Parameters": [
        "Air Temp",
        "Process Temp",
        "RPM",
        "Torque",
        "Tool Wear"
    ],
    "Values": [
        air_temp,
        process_temp,
        rpm,
        torque,
        tool_wear
    ]
})

st.bar_chart(chart_data.set_index("Parameters"))

# -------------------------------
# Prediction Section
# -------------------------------
if st.button("Predict"):

    input_data = pd.DataFrame([[
        mapping[machine_type],
        air_temp,
        process_temp,
        rpm,
        torque,
        tool_wear
    ]], columns=[
        "Type",
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]"
    ])

    # Prediction
    prediction = model.predict(input_data)

    # Confidence Score
    probability = model.predict_proba(input_data)

    st.subheader("🔍 Prediction Result")

    if prediction[0] == 1:
        st.error("⚠ Machine Failure Predicted")

        st.write(
            "Failure Probability:",
            round(probability[0][1] * 100, 2),
            "%"
        )

    else:
        st.success("✅ Machine Healthy")

        st.write(
            "Healthy Probability:",
            round(probability[0][0] * 100, 2),
            "%"
        )

# -------------------------------
# Feature Importance
# -------------------------------
st.subheader("📈 Feature Importance")

features = [
    "Type",
    "Air Temp",
    "Process Temp",
    "RPM",
    "Torque",
    "Tool Wear"
]

importance = model.feature_importances_

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": importance
})

fig, ax = plt.subplots()

ax.bar(
    importance_df["Feature"],
    importance_df["Importance"]
)

plt.xticks(rotation=45)
plt.title("Feature Importance in Prediction")

st.pyplot(fig)