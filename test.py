import joblib
import numpy as np
import pandas as pd

# Load the saved model, scaler, and feature columns
model = joblib.load('xgb_valuation_model.pkl')
scaler = joblib.load('scaler_valuation.pkl')
feature_columns = joblib.load('feature_columns.pkl')

def preprocess_input(user_input):
    """
    Preprocess the user input to match the model's expected input format.
    """
    # Convert user input to DataFrame
    input_df = pd.DataFrame([user_input])
    
    # Feature Engineering
    current_year = 2025
    input_df['Funding Efficiency Ratio'] = input_df['Investment Amount (USD)'] / input_df['Number of Investors']
    input_df['Funding Rounds Per Year'] = input_df['Funding Rounds'] / (current_year - input_df['Year Founded'] + 1)
    input_df['Investment Per Round'] = input_df['Investment Amount (USD)'] / input_df['Funding Rounds']
    input_df['Investor Density'] = input_df['Number of Investors'] / input_df['Funding Rounds']
    input_df['Startup Age'] = current_year - input_df['Year Founded']
    input_df['Early_Late_Stage'] = (input_df['Funding Rounds'] >= 5).astype(int)
    
    # One-Hot Encode categorical columns
    input_df_encoded = pd.get_dummies(input_df, columns=['Industry', 'Country'], drop_first=True)
    
    # Ensure all feature columns are present
    for col in feature_columns:
        if col not in input_df_encoded.columns:
            input_df_encoded[col] = 0
    
    # Reorder columns to match the training data
    input_df_encoded = input_df_encoded[feature_columns]
    
    # Normalize numerical columns
    input_df_encoded[['Investment Amount (USD)']] = scaler.transform(input_df_encoded[['Investment Amount (USD)']])
    
    return input_df_encoded

def validate_model(user_input):
    """
    Validate the model based on user input.
    """
    # Preprocess the input
    input_data = preprocess_input(user_input)
    
    # Make prediction
    prediction = model.predict(input_data)
    
    # Convert log-transformed prediction back to original scale
    valuation_prediction = np.expm1(prediction[0])
    
    return valuation_prediction

# Example user input
user_input = {
    'Funding Rounds': 3,
    'Year Founded': 2018,
    'Number of Investors': 10,
    'Investment Amount (USD)': 5000000,
    'Industry': 'Technology',
    'Country': 'USA'
}

# Validate the model with the example input
predicted_valuation = validate_model(user_input)
print(f"Predicted Valuation (USD): {predicted_valuation:.2f}")