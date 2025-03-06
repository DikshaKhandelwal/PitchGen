import streamlit as st
import pandas as pd
import plotly.express as px

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("./investor/startup_funding.csv", encoding="ISO-8859-1")

    # Rename columns for consistency
    df.rename(columns={
        "Startup Name": "Business Name",
        "Investors Name": "Investor",
        "Amount in USD": "InvestmentAmount in USD",
        "Industry Vertical": "Industry"
    }, inplace=True)

    # Drop missing values in relevant columns
    df.dropna(subset=["Investor", "Industry", "InvestmentAmount in USD"], inplace=True)

    # Convert Amount column to numeric (removing non-numeric characters)
    df["InvestmentAmount in USD"] = df["InvestmentAmount in USD"].astype(str).str.replace(",", "").str.extract("(\d+)").astype(float)

    # Split multiple investors into separate rows
    df["Investor"] = df["Investor"].str.split(",")
    df = df.explode("Investor")

    # Clean investor, industry, and business names
    df["Investor"] = df["Investor"].str.strip().str.lower()
    df["Industry"] = df["Industry"].str.strip().str.lower()
    df["Business Name"] = df["Business Name"].str.strip().str.lower()

    return df

def show():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Lora:wght@400;500;600;700&display=swap');

        body {
            font-family: 'Lora', serif;
            font-size: 18px;
        }
    </style>
    """, unsafe_allow_html=True)
    df = load_data()

    # Streamlit UI
    st.title("Investor Lookup")
    st.write("Search investors by **Industry** or **Business Name**.")

    # Choose search type
    search_type = st.radio("Search by:", ["Industry", "Business Name"])

    if "search_results" not in st.session_state:
        st.session_state.search_results = []  # Initialize storage

    def store_top_10_results(search_type, search_query, result_df):
        """Stores only the top 10 results in session state."""
        top_10_results = result_df.head(10).to_dict()
        st.session_state.search_results = [{
            "Search Type": search_type,
            "Search Query": search_query,
            "Results": top_10_results
        }]

    if search_type == "Industry":
        industry_name = st.text_input("Enter Industry Name:", "").strip().lower()

        def get_investor_details_by_industry(industry_name):
            if not industry_name:
                return None
            data = df[df["Industry"] == industry_name][["Investor", "Business Name", "InvestmentAmount in USD"]]
            return data.sort_values(by="InvestmentAmount in USD", ascending=False).reset_index(drop=True) if not data.empty else None

        if st.button("Find Investors"):
            result_df = get_investor_details_by_industry(industry_name)
            if result_df is not None:
                result_df = result_df.head(10)  # Limit to top 10
                st.success(f"### Top 10 Investors in **{industry_name.capitalize()}** Industry")
                st.dataframe(result_df)

                store_top_10_results("Industry", industry_name, result_df)

                # Plot top investors
                fig = px.bar(result_df, x="Investor", y="InvestmentAmount in USD", title="Top 10 Investors")
                st.plotly_chart(fig)
            else:
                st.warning("No investors found for this industry.")

    elif search_type == "Business Name":
        business_name = st.text_input("Enter Business Name:", "").strip().lower()

        def get_investor_details_by_business(business_name):
            if not business_name:
                return None
            data = df[df["Business Name"] == business_name][["Investor", "Industry", "InvestmentAmount in USD"]]
            return data.sort_values(by="InvestmentAmount in USD", ascending=False).reset_index(drop=True) if not data.empty else None

        if st.button("Find Investors"):
            result_df = get_investor_details_by_business(business_name)
            if result_df is not None:
                result_df = result_df.head(10)  # Limit to top 10
                st.success(f"### Top 10 Investors for **{business_name.capitalize()}**")
                st.dataframe(result_df)

                store_top_10_results("Business Name", business_name, result_df)
            else:
                st.warning("No investors found for this business.")

    # Display previous search (only the latest top 10 results)
    if st.session_state.search_results:
        st.subheader("Last Search Results (Top 10)")
        last_search = st.session_state.search_results[0]
        st.write(f"**{last_search['Search Type']} Search for:** {last_search['Search Query'].capitalize()}")
        st.dataframe(pd.DataFrame(last_search["Results"]))
