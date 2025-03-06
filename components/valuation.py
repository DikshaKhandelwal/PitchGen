import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# Load model and scaler
model = joblib.load("./xgb_valuation_model2.pkl")
scaler = joblib.load("./scaler_valuation2.pkl")
feature_columns = joblib.load("./feature_columns2.pkl")

# Initialize session state for history
if "valuation_history" not in st.session_state:
    st.session_state.valuation_history = []

def show():
    # Add custom styling
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Lora:wght@400;500;600;700&display=swap');

        body {
            font-family: 'Lora', serif;
            font-size: 18px;
        }

        .stButton > button {
            background-color: #1b096b;
            color: white;
            border-radius: 4px;
            border: none;
            padding: 12px 24px;
            font-weight: 500;
            font-size: 18px;
        }
        .stButton > button:hover {
            background-color: #1e0880 !important;
            color: white !important;
            border: 1px solid white !important;
        }
        .stSelectbox > div > div {
            border-radius: 4px;
            background-color: #f1f5f9;
        }
        .stNumberInput > div > div > input {
            border-radius: 4px;
            background-color: #f1f5f9;
        }
        div[data-testid="stTable"] {
            background-color: white;
            border-radius: 8px;
            padding: 10px;
            border: 1px solid #e5e7eb;
        }
        div[data-testid="stTable"] td {
            color: #1e3a8a;
            font-size: 18px;
        }
        div.stSuccess {
            background-color: #dbeafe;
            color: #1e3a8a;
            border: none;
            padding: 20px;
            border-radius: 8px;
            font-size: 18px;
        }
    </style>
    """, unsafe_allow_html=True)

    # Header with styled container
    st.markdown("""
    <div style="background-color: white; padding: 20px; border-radius: 8px; border: 1px solid #e5e7eb; margin-bottom: 20px;">
        <h1 style="color: #1e3a8a; font-weight: 700; margin-bottom: 5px; font-size: 32px;">💰 Startup Valuation Predictor</h1>
        <p style="color: #6b7280; font-size: 20px;">Enter your startup details to estimate the valuation.</p>
    </div>
    """, unsafe_allow_html=True)

    # Group inputs in styled containers
    with st.container():
        st.markdown('<div style="background-color: white; padding: 20px; border-radius: 8px; border: 1px solid #e5e7eb;">', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        with col1:
            funding_rounds = st.number_input("Funding Rounds", min_value=1, step=1)
            year_founded = st.number_input("Year Founded", min_value=1900, max_value=2025, step=1)
            num_investors = st.number_input("Number of Investors", min_value=1, step=1)
            investment_amount = st.number_input("Investment Amount (USD)", min_value=10000, step=10000)
        
        with col2:
            growth_rate = st.number_input("Growth Rate (%)", min_value=0.0, step=0.1) / 100
            industry = st.selectbox("Industry", ["AI", "Healthcare", "Finance", "E-commerce", "Other"])
            country = st.selectbox("Country", ["USA", "India", "UK", "Germany", "Other"])
            current_year = datetime.now().year
            target_year = st.number_input("Predict Valuation for Year", min_value=current_year, max_value=current_year + 20, value=current_year)
        
        st.markdown('</div>', unsafe_allow_html=True)

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

    # Style the prediction display
    if st.button("Predict Valuation", key="predict_btn"):
        prediction = model.predict(input_data)[0]
        valuation = np.expm1(prediction)

        st.markdown(f"""
        <div style="background-color: #dbeafe; padding: 20px; border-radius: 8px; margin: 20px 0;">
            <h2 style="color: #1e3a8a; margin-bottom: 10px;">Estimated Valuation</h2>
            <p style="color: #1e3a8a; font-size: 24px; font-weight: 600;">${valuation:,.2f}</p>
            <p style="color: #1e3a8a;">Predicted for year {target_year}</p>
        </div>
        """, unsafe_allow_html=True)

        # Save result in session state
        st.session_state.valuation_history.append({
            "Funding Rounds": funding_rounds,
            "Year Founded": year_founded,
            "Number of Investors": num_investors,
            "Investment Amount": investment_amount,
            "Growth Rate (%)": growth_rate * 100,
            "Industry": industry,
            "Country": country,
            "Target Year": target_year,
            "Predicted Valuation (USD)": f"${valuation:,.2f}"
        })

    # Style the history table
    if st.session_state.valuation_history:
        st.markdown("""
        <div style="background-color: white; padding: 20px; border-radius: 8px; border: 1px solid #e5e7eb; margin-top: 20px;">
            <h2 style="color: #1e3a8a; margin-bottom: 15px;">📊 Valuation History</h2>
        </div>
        """, unsafe_allow_html=True)
        st.table(pd.DataFrame(st.session_state.valuation_history))
