import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

# Load Data
df = pd.read_csv("investments_VC.csv", encoding="ISO-8859-1")

# Clean Column Names
df.columns = df.columns.str.strip()

# Convert Investment Amount to Numeric
df['funding_total_usd'] = df['funding_total_usd'].astype(str).str.replace(',', '').replace(' ', '').replace('-', '0')
df['funding_total_usd'] = pd.to_numeric(df['funding_total_usd'], errors='coerce')

# Convert Date Columns to Datetime
date_cols = ['founded_at', 'first_funding_at', 'last_funding_at']
for col in date_cols:
    df[col] = pd.to_datetime(df[col], errors='coerce')

# Feature Engineering
current_year = 2025
df['Startup Age'] = current_year - df['founded_at'].dt.year
df['Funding Efficiency'] = df['funding_total_usd'] / (df['funding_rounds'] + 1)

# Handle Missing Values
df = df.dropna(subset=['funding_rounds', 'Startup Age', 'funding_total_usd'])

# One-Hot Encoding for Categorical Features
df_encoded = pd.get_dummies(df, columns=['market', 'country_code'], drop_first=True)

# Feature Scaling
scaler = MinMaxScaler()
df_encoded[['funding_total_usd']] = scaler.fit_transform(df_encoded[['funding_total_usd']])

# Define Features and Target
important_features = ['Startup Age', 'funding_total_usd', 'Funding Efficiency']
X = df_encoded[important_features]
y = np.log1p(df_encoded['funding_rounds'])  # Log Transformation

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost Model
xgb_model = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=42)
xgb_model.fit(X_train, y_train)

# Predictions
y_train_pred = xgb_model.predict(X_train)
y_test_pred = xgb_model.predict(X_test)

# Convert Back from Log Scale
y_train_pred = np.expm1(y_train_pred)
y_test_pred = np.expm1(y_test_pred)
y_train_actual = np.expm1(y_train)
y_test_actual = np.expm1(y_test)

# Model Evaluation
train_mae = mean_absolute_error(y_train_actual, y_train_pred)
train_r2 = r2_score(y_train_actual, y_train_pred)
test_mae = mean_absolute_error(y_test_actual, y_test_pred)
test_r2 = r2_score(y_test_actual, y_test_pred)

# Print Metrics
print(f"🔹 Training MAE: {train_mae:.4f}")
print(f"🔹 Training R² Score: {train_r2:.4f}")
print(f"🔹 Testing MAE: {test_mae:.4f}")
print(f"🔹 Testing R² Score: {test_r2:.4f}")

# Save Model
joblib.dump(xgb_model, "xgb_startup_funding_model1.pkl")
joblib.dump(scaler, "scaler_funding1.pkl")
print("✅ Model and scaler saved!")
