import pandas as pd  

# Load dataset
df = pd.read_csv("startup_funding.csv", encoding="ISO-8859-1")

# Print actual column names (for debugging)
print("Actual Columns:", df.columns)

# Rename columns based on actual dataset
df.rename(columns={
    "Startup Name": "Business",
    "Investors Name": "Investor",
    "Amount in USD": "InvestmentAmount",
    "Industry Vertical": "Industry"
}, inplace=True)

# Drop missing values in relevant columns
df.dropna(subset=["Investor", "Business"], inplace=True)

# Split multiple investors into separate rows
df["Investor"] = df["Investor"].str.split(",")  
df = df.explode("Investor")

# Clean investor and business names
df["Investor"] = df["Investor"].str.strip().str.lower()
df["Business"] = df["Business"].str.strip().str.lower()

# Display first 10 cleaned rows
print(df.head(10))
