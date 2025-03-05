import streamlit as st
import numpy as np
import joblib
import pandas as pd

# Load the trained model and scaler
model = joblib.load("xgb_startup_funding_model1.pkl")
scaler = joblib.load("scaler_funding1.pkl")

# Function for prediction
def predict_funding_rounds(startup_age, investment_amount):
    """Predict funding rounds based on startup age and investment amount."""
    funding_efficiency = investment_amount / (startup_age + 1)
    investment_amount_scaled = scaler.transform([[investment_amount]])[0][0]

    input_data = pd.DataFrame({
        "Startup Age": [startup_age],
        "funding_total_usd": [investment_amount_scaled],
        "Funding Efficiency": [funding_efficiency]
    })

    prediction = model.predict(input_data)
    return np.expm1(prediction)[0]  # Convert from log scale

# Streamlit UI
def show():
    st.title("💰 Startup Funding Rounds Predictor")
    st.write("Enter details about your startup to predict the expected number of funding rounds.")

    startup_age = st.number_input("Startup Age (Years)", min_value=0, max_value=100, value=5)
    investment_amount = st.number_input("Total Investment Amount (USD)", min_value=0.0, value=1000000.0, step=10000.0)

    if "predictions" not in st.session_state:
        st.session_state.predictions = []  # Initialize session storage

    if st.button("Predict Funding Rounds"):
        funding_rounds = predict_funding_rounds(startup_age, investment_amount)
        
        # Store in session state (for current instance only)
        st.session_state.predictions.append({
            "Startup Age": startup_age,
            "Investment Amount": investment_amount,
            "Predicted Funding Rounds": round(funding_rounds, 2)
        })

        st.success(f"Predicted Number of Funding Rounds: {funding_rounds:.2f}")

    # Display previous predictions in the same session
    if st.session_state.predictions:
        st.subheader("Previous Predictions (This Session Only)")
        predictions_df = pd.DataFrame(st.session_state.predictions)
        st.dataframe(predictions_df)

