import streamlit as st
import numpy as np
import joblib
import pandas as pd

# Load the trained model and scaler
model = joblib.load("xgb_startup_funding_model1.pkl")
scaler = joblib.load("scaler_funding1.pkl")

def show():
    st.title("💰 Startup Funding Rounds Predictor")
    st.write("Enter details about your startup to predict the expected number of funding rounds.")

    # User Inputs
    startup_age = st.number_input("Startup Age (Years)", min_value=0, max_value=100, value=5)
    investment_amount = st.number_input("Total Investment Amount (USD)", min_value=0.0, value=1000000.0, step=10000.0)

    # Compute Funding Efficiency
    funding_efficiency = investment_amount / (startup_age + 1)

    # Normalize Investment Amount
    investment_amount_scaled = scaler.transform([[investment_amount]])[0][0]

    # Create Input DataFrame
    input_data = pd.DataFrame({
        "Startup Age": [startup_age],
        "funding_total_usd": [investment_amount_scaled],
        "Funding Efficiency": [funding_efficiency]
    })

    if st.button("Predict Funding Rounds"):
        prediction = model.predict(input_data)
        funding_rounds = np.expm1(prediction)[0]  # Convert from log scale
        st.success(f"Predicted Number of Funding Rounds: {funding_rounds:.2f}")
        return funding_rounds
    return 0.0