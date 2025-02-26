

import pandas as pd
import numpy as np



df = pd.read_csv("startup_growth_investment_data.csv")

df.info()

df.head()

import matplotlib.pyplot as plt
import seaborn as sns

# Set the style for seaborn plots
sns.set_style("whitegrid")

# Create scatter plots to visualize the impact of different factors

# Investment Amount vs Valuation
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x="Investment Amount (USD)", y="Valuation (USD)", alpha=0.6)
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Investment Amount (USD) [Log Scale]")
plt.ylabel("Valuation (USD) [Log Scale]")
plt.title("Investment Amount vs Valuation")
plt.show()

# Number of Investors vs Valuation
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x="Number of Investors", y="Valuation (USD)", alpha=0.6)
plt.yscale("log")
plt.xlabel("Number of Investors")
plt.ylabel("Valuation (USD) [Log Scale]")
plt.title("Number of Investors vs Valuation")
plt.show()

# Funding Rounds vs Investment Amount
plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x="Funding Rounds", y="Investment Amount (USD)")
plt.yscale("log")
plt.xlabel("Funding Rounds")
plt.ylabel("Investment Amount (USD) [Log Scale]")
plt.title("Funding Rounds vs Investment Amount")
plt.show()

# Industry-wise Investment Amount Distribution
plt.figure(figsize=(10, 6))
top_industries = df.groupby("Industry")["Investment Amount (USD)"].median().sort_values(ascending=False).head(10).index
sns.boxplot(data=df[df["Industry"].isin(top_industries)], x="Industry", y="Investment Amount (USD)")
plt.xticks(rotation=45)
plt.yscale("log")
plt.xlabel("Industry")
plt.ylabel("Investment Amount (USD) [Log Scale]")
plt.title("Investment Amount Distribution by Industry")
plt.show()

df.shape

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, r2_score

# Define features and target variable
X = df.drop(columns=["Startup Name", "Valuation (USD)"])  # Exclude non-numeric features
y = df["Valuation (USD)"]

# Identify categorical and numerical features
categorical_features = ["Industry", "Country"]
numerical_features = ["Funding Rounds", "Investment Amount (USD)", "Number of Investors", "Year Founded", "Growth Rate (%)"]

# Preprocessing pipeline
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numerical_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
])

# Define model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Get feature importance
feature_importance = model.named_steps["regressor"].feature_importances_

# Extract feature names after transformation
feature_names = numerical_features + list(model.named_steps["preprocessor"].named_transformers_["cat"].get_feature_names_out(categorical_features))

# Create a DataFrame for feature importance
importance_df = pd.DataFrame({"Feature": feature_names, "Importance": feature_importance}).sort_values(by="Importance", ascending=False)

# Display feature importance in a readable format
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
importance_df.sort_values(by="Importance", ascending=True).plot(
    kind="barh", x="Feature", y="Importance", legend=False
)
print(importance_df.sort_values(by="Importance", ascending=True))
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.title("Feature Importance for Startup Valuation Prediction")
plt.show()

# Print evaluation metrics
mae, r2

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

# Load dataset
file_path = "startup_growth_investment_data.csv"
df_growth = pd.read_csv(file_path)

# Drop 'Startup Name' only if it exists
if 'Startup Name' in df_growth.columns:
    df_growth = df_growth.drop(columns=['Startup Name'])

# Feature Engineering
current_year = 2025  # Adjust this based on the latest year of your dataset
df_growth['Funding Efficiency Ratio'] = df_growth['Investment Amount (USD)'] / df_growth['Number of Investors']
df_growth['Funding Rounds Per Year'] = df_growth['Funding Rounds'] / (current_year - df_growth['Year Founded'] + 1)
df_growth['Investment Per Round'] = df_growth['Investment Amount (USD)'] / df_growth['Funding Rounds']
df_growth['Investor Density'] = df_growth['Number of Investors'] / df_growth['Funding Rounds']
df_growth['Startup Age'] = current_year - df_growth['Year Founded']

df_growth['Early_Late_Stage'] = (df_growth['Funding Rounds'] >= 5).astype(int)  # Example threshold for early vs late-stage

# Compute Industry-Specific Growth Multipliers
industry_median_growth = df_growth.groupby('Industry')['Growth Rate (%)'].median()
df_growth['Industry Growth Multiplier'] = df_growth['Industry'].map(industry_median_growth)
df_growth['Growth Rate Relative to Industry'] = df_growth['Growth Rate (%)'] / df_growth['Industry Growth Multiplier']

# Compute Country-Specific Investment Index
country_median_investment = df_growth.groupby('Country')['Investment Amount (USD)'].median()
df_growth['Country Investment Index'] = df_growth['Country'].map(country_median_investment)
df_growth['Investment Relative to Country'] = df_growth['Investment Amount (USD)'] / df_growth['Country Investment Index']

# One-Hot Encode categorical columns
df_growth_encoded = pd.get_dummies(df_growth, columns=['Industry', 'Country'], drop_first=True)

# Normalize numerical columns (Investment, Valuation)
scaler_growth = MinMaxScaler()
df_growth_encoded[['Investment Amount (USD)', 'Valuation (USD)']] = scaler_growth.fit_transform(
    df_growth_encoded[['Investment Amount (USD)', 'Valuation (USD)']])

# Select important features based on feature importance
important_features_growth = [
    'Funding Rounds', 'Year Founded', 'Number of Investors', 'Investment Amount (USD)',
    'Funding Efficiency Ratio', 'Funding Rounds Per Year', 'Investment Per Round',
    'Investor Density', 'Startup Age', 'Early_Late_Stage', 'Growth Rate Relative to Industry',
    'Investment Relative to Country'
]
X_growth = df_growth_encoded[important_features_growth]
y_growth = df_growth_encoded['Growth Rate (%)']

# Split data into Training (80%) and Testing (20%)
X_train_growth, X_test_growth, y_train_growth, y_test_growth = train_test_split(X_growth, y_growth, test_size=0.2, random_state=42)

# Train Optimized XGBoost Model
xgb_growth_model = XGBRegressor(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42)

xgb_growth_model.fit(X_train_growth, y_train_growth)

# Predictions
y_pred_xgb_growth = xgb_growth_model.predict(X_test_growth)

# Predictions on training data
y_train_pred_xgb_growth = xgb_growth_model.predict(X_train_growth)

# Training metrics


# Training Mean Absolute Percentage Error (MAPE)
mape_train_xgb_growth = np.mean(np.abs((y_train_growth - y_train_pred_xgb_growth) / y_train_growth)) * 100
mean_accuracy_train_xgb_growth = 100 - mape_train_xgb_growth

print(f"Training Mean Absolute Error (Growth Rate): {mean_absolute_error(y_train_growth, y_train_pred_xgb_growth)}")
print(f"Training R² Score (Growth Rate): {r2_score(y_train_growth, y_train_pred_xgb_growth)}")
print(f"Training Mean Accuracy: {mean_accuracy_train_xgb_growth:.2f}%")


# Evaluate model
mae_xgb_growth = mean_absolute_error(y_test_growth, y_pred_xgb_growth)
r2_xgb_growth = r2_score(y_test_growth, y_pred_xgb_growth)

mape_xgb_growth = np.mean(np.abs((y_test_growth - y_pred_xgb_growth) / y_test_growth)) * 100
mean_accuracy_xgb_growth = 100 - mape_xgb_growth

print(f"Testing Mean Absolute Error (Growth Rate): {mae_xgb_growth}")
print(f"Testing R² Score (Growth Rate): {r2_xgb_growth}")
print(f"Testing Mean Accuracy: {mean_accuracy_xgb_growth:.2f}%")

import joblib
joblib.dump(xgb_growth_model, "xgb_startup_growth_model.pkl")