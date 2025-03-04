import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# Load model and scaler
model = joblib.load("./xgb_valuation_model2.pkl")
scaler = joblib.load("./scaler_valuation2.pkl")
feature_columns = joblib.load("./feature_columns2.pkl")

def show():
    st.title("💰 Startup Valuation Predictor")
    st.write("Enter your startup details to estimate the valuation.")

    # User inputs
    funding_rounds = st.number_input("Funding Rounds", min_value=1, step=1)
    year_founded = st.number_input("Year Founded", min_value=1900, max_value=2025, step=1)
    num_investors = st.number_input("Number of Investors", min_value=1, step=1)
    investment_amount = st.number_input("Investment Amount (USD)", min_value=10000, step=10000)
    growth_rate = st.number_input("Growth Rate (%)", min_value=0.0, step=0.1) / 100  # Convert to decimal format
    industry = st.selectbox("Industry", ["AI", "Healthcare", "Finance", "E-commerce", "Other"])
    country = st.selectbox("Country", ["USA", "India", "UK", "Germany", "Other"])

    # Select prediction year (up to 20 years ahead)
    current_year = datetime.now().year
    target_year = st.number_input("Predict Valuation for Year", min_value=current_year, max_value=current_year + 20, value=current_year)

    # Compute additional features based on target year
    startup_age = target_year - year_founded
    funding_rounds_per_year = funding_rounds / (startup_age + 1)  # Avoid division by zero
    funding_efficiency = investment_amount / num_investors
    investment_per_round = investment_amount / funding_rounds
    investor_density = num_investors / funding_rounds
    early_late_stage = int(funding_rounds >= 5)

    # Load industry and country median values from training data
    industry_median_valuation = pd.read_csv("startup_growth_investment_data.csv").groupby('Industry')['Valuation (USD)'].median()
    country_median_investment = pd.read_csv("startup_growth_investment_data.csv").groupby('Country')['Investment Amount (USD)'].median()

    industry_valuation_multiplier = industry_median_valuation.get(industry, industry_median_valuation.median())
    valuation_relative_to_industry = investment_amount / industry_valuation_multiplier

    country_investment_index = country_median_investment.get(country, country_median_investment.median())
    investment_relative_to_country = investment_amount / country_investment_index

    # Prepare input DataFrame
    input_data = pd.DataFrame([[
        funding_rounds, year_founded, num_investors, investment_amount,
        growth_rate, funding_efficiency, funding_rounds_per_year,
        investment_per_round, investor_density, startup_age, early_late_stage,
        valuation_relative_to_industry, investment_relative_to_country,
        industry, country
    ]], columns=[
        'Funding Rounds', 'Year Founded', 'Number of Investors', 'Investment Amount (USD)',
        'Growth Rate', 'Funding Efficiency Ratio', 'Funding Rounds Per Year',
        'Investment Per Round', 'Investor Density', 'Startup Age', 'Early_Late_Stage',
        'Valuation Relative to Industry', 'Investment Relative to Country',
        'Industry', 'Country'
    ])

    # One-Hot Encoding for Industry and Country
    input_data = pd.get_dummies(input_data, columns=['Industry', 'Country'], drop_first=True)

    # Ensure all columns match training features
    for col in feature_columns:
        if col not in input_data:
            input_data[col] = 0  # Add missing columns as 0

    # Keep only required feature columns
    input_data = input_data[feature_columns]

    # Normalize investment amount
    input_data[['Investment Amount (USD)']] = scaler.transform(input_data[['Investment Amount (USD)']])

    # Predict valuation
    if st.button("Predict Valuation"):
        prediction = model.predict(input_data)[0]
        valuation = np.expm1(prediction)  # Convert log valuation back to original scale
        st.success(f"Estimated Valuation in {target_year}: ${valuation:,.2f}")
        return valuation
    return 0.0
