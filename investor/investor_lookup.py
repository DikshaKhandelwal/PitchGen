import pandas as pd

# Load dataset
df = pd.read_csv("startup_funding.csv", encoding="ISO-8859-1")

# Rename columns for consistency
df.rename(columns={
    "Startup Name": "Business",
    "Investors Name": "Investor",
    "Amount in USD": "InvestmentAmount",
    "Industry Vertical": "Industry"
}, inplace=True)

# Drop missing values in relevant columns
df.dropna(subset=["Investor", "Industry"], inplace=True)

# Split multiple investors into separate rows
df["Investor"] = df["Investor"].str.split(",")  
df = df.explode("Investor")

# Clean investor and industry names
df["Investor"] = df["Investor"].str.strip().str.lower()
df["Industry"] = df["Industry"].str.strip().str.lower()

# Function to fetch investors by industry
def get_investors_by_industry(industry_name):
    """
    Fetch investors who have invested in a given industry.
    
    :param industry_name: Name of the industry (case insensitive)
    :return: List of unique investors
    """
    industry_name = industry_name.lower().strip()  # Normalize input
    investors = df[df["Industry"] == industry_name]["Investor"].unique()

    if len(investors) == 0:
        return "No investors found for this industry."
    
    return list(investors)

# Example usage
industry_name = "e-commerce"  # Change this to test with other industries
investors_list = get_investors_by_industry(industry_name)
print(f"Investors in {industry_name}: {investors_list}")
