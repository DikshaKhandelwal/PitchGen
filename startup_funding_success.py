
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score,classification_report
from xgboost import XGBRegressor
from IPython.display import Image


import six
import sys
sys.modules['sklearn.externals.six'] = six
from sklearn.externals.six import StringIO

from pydot import graph_from_dot_data
from sklearn.utils import resample



from sklearn.tree import export_graphviz
import pydot
import random

df = pd.read_csv("investments_VC.csv",encoding="ISO-8859-1")

df.info()

df.head()

df.shape

print("\nMissing Values:")
print(df.isnull().sum())

df = df.rename(columns={' market ': "market", ' funding_total_usd ': "funding_total_usd"})

df['funding_total_usd'] = df['funding_total_usd'].astype(str).str.replace(',', '')
df['funding_total_usd'] = df['funding_total_usd'].str.replace(' ', '')
df['funding_total_usd'] = df['funding_total_usd'].str.replace('-', '0')
df['funding_total_usd'] = pd.to_numeric(df['funding_total_usd'], errors='coerce')  # Converts to numeric, handling errors

df = df.rename(columns=lambda x: x.strip())  # Removes extra spaces from column names
df.columns = df.columns.str.strip()  # Removes leading/trailing spaces from all column names
print(df.columns)
#df['founded_at'] =  pd.to_datetime(df['founded_at'], format='%Y-%m-%d', errors = 'coerce') # conveting column into date and ignoring errors
df['founded_at'] = pd.to_datetime(df['founded_at'], format='%Y-%m-%d', errors='coerce')

df['first_funding_at'] = pd.to_datetime(df['first_funding_at'], format='%Y-%m-%d', errors='coerce')
df['last_funding_at'] = pd.to_datetime(df['last_funding_at'], format='%Y-%m-%d', errors='coerce')
df['founded_year'] = pd.to_datetime(df['founded_year'], format='%Y', errors='coerce')
df['founded_month'] = pd.to_datetime(df['founded_month'], format='%Y-%m', errors='coerce')
df['market'] = df['market'].astype(str).str.strip()

# Drop 'Startup Name' if it exists
if 'Startup Name' in df.columns:
    df = df.drop(columns=['Startup Name'])

# Data Visualization
plt.figure(figsize=(10, 5))
sns.histplot(df['funding_total_usd'], bins=30, kde=True)
plt.title("Distribution of Investment Amount")
plt.show()

# Feature Engineering
current_year = 2025
df['Startup Age'] = current_year - df['founded_at'].dt.year

df_encoded = pd.get_dummies(df, columns=['market', 'country_code'], drop_first=True)
scaler = MinMaxScaler()
df_encoded[['funding_total_usd']] = scaler.fit_transform(df_encoded[['funding_total_usd']])

important_features = ['funding_rounds', 'Startup Age', 'funding_total_usd']
X = df_encoded[important_features]
y = df_encoded['funding_rounds']



df['Startup Age'] = current_year - df['founded_at'].dt.year

df_encoded = pd.get_dummies(df, columns=['market', 'country_code'], drop_first=True)
scaler = MinMaxScaler()
df_encoded[['funding_total_usd']] = scaler.fit_transform(df_encoded[['funding_total_usd']])

important_features = ['funding_rounds', 'Startup Age', 'funding_total_usd']
X = df_encoded[important_features]
y = pd.to_numeric(df_encoded['funding_rounds'], errors='coerce')

# Remove NaN and infinite values from y
y.replace([np.inf, -np.inf], np.nan, inplace=True)
y.dropna(inplace=True)

# Ensure X and y have the same length
valid_indices = y.dropna().index
X = X.loc[valid_indices]
y = y.loc[valid_indices]

# Split data into Training (80%) and Testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"NaN values in y_train: {y_train.isna().sum()}")
print(f"Any Inf values in y_train: {np.isinf(y_train).sum()}")

# Train Optimized XGBoost Model
xgb_model = XGBRegressor(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42)

xgb_model.fit(X_train, y_train)

# Predictions
y_pred_xgb = xgb_model.predict(X_test)


# Evaluate model on Testing Data
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
r2_xgb = r2_score(y_test, y_pred_xgb)
print(f"Testing Mean Absolute Error: {mae_xgb}")
print(f"Testing R² Score: {r2_xgb}")

# Evaluate model on Training Data
y_train_pred = xgb_model.predict(X_train)
mae_train = mean_absolute_error(y_train, y_train_pred)
r2_train = r2_score(y_train, y_train_pred)
print(f"Training Mean Absolute Error: {mae_train}")
print(f"Training R² Score: {r2_train}")

# Calculate Accuracy
train_accuracy = np.mean(np.round(y_train_pred) == y_train)
test_accuracy = np.mean(np.round(y_pred_xgb) == y_test)
print(f"Training Accuracy: {train_accuracy}")
print(f"Testing Accuracy: {test_accuracy}")

# Classification Report
y_test_rounded = np.round(y_test)
y_pred_rounded = np.round(y_pred_xgb)
print("Classification Report (Testing):")
print(classification_report(y_test_rounded, y_pred_rounded))

import joblib
joblib.dump(xgb_model, "xgb_startup_funding_success_model.pkl") 
