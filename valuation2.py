import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor
import joblib

# Load dataset
file_path = "startup_growth_investment_data.csv"
df = pd.read_csv(file_path)

# Drop 'Startup Name' if it exists
if 'Startup Name' in df.columns:
    df = df.drop(columns=['Startup Name'])

# Define current year
current_year = 2025  

# Convert Growth Rate from percentage to decimal format
df['Growth Rate'] = df['Growth Rate (%)'] / 100.0  

# Feature Engineering
df['Funding Efficiency Ratio'] = df['Investment Amount (USD)'] / df['Number of Investors']
df['Funding Rounds Per Year'] = df['Funding Rounds'] / (current_year - df['Year Founded'] + 1)
df['Investment Per Round'] = df['Investment Amount (USD)'] / df['Funding Rounds']
df['Investor Density'] = df['Number of Investors'] / df['Funding Rounds']
df['Startup Age'] = current_year - df['Year Founded']
df['Early_Late_Stage'] = (df['Funding Rounds'] >= 5).astype(int)  

# Compute Industry-Specific Multipliers
industry_median_valuation = df.groupby('Industry')['Valuation (USD)'].median()
df['Industry Valuation Multiplier'] = df['Industry'].map(industry_median_valuation)
df['Valuation Relative to Industry'] = df['Valuation (USD)'] / df['Industry Valuation Multiplier']

# Compute Country-Specific Investment Index
country_median_investment = df.groupby('Country')['Investment Amount (USD)'].median()
df['Country Investment Index'] = df['Country'].map(country_median_investment)
df['Investment Relative to Country'] = df['Investment Amount (USD)'] / df['Country Investment Index']

# One-Hot Encode categorical columns
df_encoded = pd.get_dummies(df, columns=['Industry', 'Country'], drop_first=True)

# Log Transformation to Normalize Valuation
df_encoded['Valuation (USD)'] = np.log1p(df_encoded['Valuation (USD)'])

# Normalize numerical columns
scaler = MinMaxScaler()
df_encoded[['Investment Amount (USD)']] = scaler.fit_transform(df_encoded[['Investment Amount (USD)']])

# Select Features
features = [
    'Funding Rounds', 'Year Founded', 'Number of Investors', 'Investment Amount (USD)',
    'Growth Rate',  # ✅ Added Growth Rate
    'Funding Efficiency Ratio', 'Funding Rounds Per Year', 'Investment Per Round',
    'Investor Density', 'Startup Age', 'Early_Late_Stage', 'Valuation Relative to Industry',
    'Investment Relative to Country'
]

# Include all one-hot encoded categorical features
features += [col for col in df_encoded.columns if col.startswith(('Industry_', 'Country_'))]

# Prepare Data
X = df_encoded[features]
y = df_encoded['Valuation (USD)']  # Target: Log-transformed Valuation

# Train-Test Split (80-20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter Optimization using RandomizedSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [4, 6, 8],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}

xgb_model = XGBRegressor(random_state=42)
random_search = RandomizedSearchCV(xgb_model, param_grid, cv=5, n_iter=10, scoring='r2', n_jobs=-1, random_state=42)
random_search.fit(X_train, y_train)

# Best model
best_xgb = random_search.best_estimator_

# Predictions
y_train_pred = best_xgb.predict(X_train)
y_test_pred = best_xgb.predict(X_test)

# Training Metrics
train_mae = mean_absolute_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)
train_mape = np.mean(np.abs((y_train - y_train_pred) / (y_train + 1e-10))) * 100
train_accuracy = 100 - train_mape

# Testing Metrics
test_mae = mean_absolute_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)
test_mape = np.mean(np.abs((y_test - y_test_pred) / (y_test + 1e-10))) * 100
test_accuracy = 100 - test_mape

# Print Results
print(f"🔹 Training MAE: {train_mae:.4f}")
print(f"🔹 Training R² Score: {train_r2:.4f}")
print(f"🔹 Training Mean Accuracy: {train_accuracy:.2f}%\n")

print(f"🔹 Testing MAE: {test_mae:.4f}")
print(f"🔹 Testing R² Score: {test_r2:.4f}")
print(f"🔹 Testing Mean Accuracy: {test_accuracy:.2f}%")

# Save Feature Names, Model, and Scaler
joblib.dump(X.columns.tolist(), "feature_columns2.pkl")
joblib.dump(best_xgb, 'xgb_valuation_model2.pkl')
joblib.dump(scaler, 'scaler_valuation2.pkl')

print("✅ Model and scaler saved successfully!")
