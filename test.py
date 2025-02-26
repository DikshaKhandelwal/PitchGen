import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor

# Load trained model, scaler, and feature names
model = joblib.load('xgb_valuation_model2.pkl')
scaler = joblib.load('scaler_valuation2.pkl')
feature_columns = joblib.load("feature_columns2.pkl")

# Define current year
current_year = 2025

# Function to predict valuation
def predict_valuation(startup_data):
    # Convert Growth Rate to decimal
    startup_data['Growth Rate'] = startup_data['Growth Rate (%)'] / 100.0

    # Feature Engineering
    startup_data['Funding Efficiency Ratio'] = startup_data['Investment Amount (USD)'] / startup_data['Number of Investors']
    startup_data['Funding Rounds Per Year'] = startup_data['Funding Rounds'] / (current_year - startup_data['Year Founded'] + 1)
    startup_data['Investment Per Round'] = startup_data['Investment Amount (USD)'] / startup_data['Funding Rounds']
    startup_data['Investor Density'] = startup_data['Number of Investors'] / startup_data['Funding Rounds']
    startup_data['Startup Age'] = current_year - startup_data['Year Founded']
    startup_data['Early_Late_Stage'] = 1 if startup_data['Funding Rounds'] >= 5 else 0

    # Load industry and country median values from training data
    industry_median_valuation = pd.read_csv("startup_growth_investment_data.csv").groupby('Industry')['Valuation (USD)'].median()
    country_median_investment = pd.read_csv("startup_growth_investment_data.csv").groupby('Country')['Investment Amount (USD)'].median()

    startup_data['Industry Valuation Multiplier'] = industry_median_valuation.get(startup_data['Industry'], industry_median_valuation.median())
    startup_data['Valuation Relative to Industry'] = startup_data['Investment Amount (USD)'] / startup_data['Industry Valuation Multiplier']

    startup_data['Country Investment Index'] = country_median_investment.get(startup_data['Country'], country_median_investment.median())
    startup_data['Investment Relative to Country'] = startup_data['Investment Amount (USD)'] / startup_data['Country Investment Index']

    # Prepare DataFrame for prediction
    input_df = pd.DataFrame([startup_data])

    # One-Hot Encoding for Industry and Country
    input_df = pd.get_dummies(input_df, columns=['Industry', 'Country'], drop_first=True)

    # Ensure all columns match training features
    for col in feature_columns:
        if col not in input_df:
            input_df[col] = 0  # Add missing columns as 0

    # Keep only required feature columns
    input_df = input_df[feature_columns]

    # Scale investment amount
    input_df[['Investment Amount (USD)']] = scaler.transform(input_df[['Investment Amount (USD)']])

    # Predict using trained model
    log_valuation_pred = model.predict(input_df)
    valuation_pred = np.expm1(log_valuation_pred)  # Convert log-scale prediction back to normal

    return valuation_pred[0]

# Example User Input
startup_data = {
    "Funding Rounds": 4,
    "Year Founded": 2021,
    "Number of Investors": 24,
    "Investment Amount (USD)": 572845983.87,
    "Growth Rate (%)": 110.9,  # Now included!
    "Industry": "AI",
    "Country": "USA"
}

# Get prediction
predicted_valuation = predict_valuation(startup_data)

print(f"💰 Predicted Valuation (USD): {predicted_valuation:,.2f}")
