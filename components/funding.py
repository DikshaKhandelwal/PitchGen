import streamlit as st
import numpy as np
import joblib
import pandas as pd

# Load the trained model and scaler
model = joblib.load("xgb_startup_funding_model1.pkl")
scaler = joblib.load("scaler_funding1.pkl")

# Function for prediction
def predict_funding_rounds(startup_age, investment_amount, funding_efficiency):
    """Predict funding rounds based on startup age, investment amount, and funding efficiency."""
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
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Lora:wght@400;500;600;700&display=swap');

        body {
            font-family: 'Lora', serif;
            font-size: 18px;
        }
    </style>
    """, unsafe_allow_html=True)
    st.title("💰 Startup Funding Rounds Predictor")
    st.write("Enter details about your startup to predict the expected number of funding rounds.")

    col1, col2 = st.columns(2)
    
    with col1:
        startup_age = st.number_input("Startup Age (Years)", min_value=0, max_value=100, value=5)
        investment_amount = st.number_input("Total Investment Amount (USD)", 
                                          min_value=0.0, value=1000000.0, 
                                          step=10000.0,
                                          format="%.2f")
    
    with col2:
        funding_efficiency = st.number_input("Funding Efficiency (USD/Year)", 
                                           min_value=0.0,
                                           value=200000.0,
                                           step=10000.0,
                                           help="Average amount raised per year of operation",
                                           format="%.2f")
        
        # Display calculated metrics
        if startup_age > 0:
            actual_efficiency = investment_amount / startup_age
            st.info(f"Your actual funding efficiency: ${actual_efficiency:,.2f}/year")

    if "predictions" not in st.session_state:
        st.session_state.predictions = []  # Initialize session storage

    if st.button("Predict Funding Rounds"):
        funding_rounds = predict_funding_rounds(startup_age, investment_amount, funding_efficiency)
        
        # Store in session state (for current instance only)
        st.session_state.predictions.append({
            "Startup Age": startup_age,
            "Investment Amount": f"${investment_amount:,.2f}",
            "Funding Efficiency": f"${funding_efficiency:,.2f}/year",
            "Predicted Funding Rounds": f"{funding_rounds:.2f}"
        })

        st.markdown(f"""
        <div style="background-color: #dbeafe; padding: 20px; border-radius: 8px; margin: 20px 0;">
            <h2 style="color: #1e3a8a; margin-bottom: 10px;">Prediction Results</h2>
            <p style="color: #1e3a8a; font-size: 24px; font-weight: 600;">
                Predicted Number of Funding Rounds: {funding_rounds:.2f}
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Display previous predictions in the same session
    if st.session_state.predictions:
        st.subheader("Previous Predictions (This Session Only)")
        predictions_df = pd.DataFrame(st.session_state.predictions)
        st.dataframe(predictions_df)

