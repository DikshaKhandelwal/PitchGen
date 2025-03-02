import streamlit as st
import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import re

# Function to load and merge all sentiment analysis files
def load_all_sentiment_files():
    files = glob.glob("*_finbert_output.csv")  # Find all relevant files
    if not files:
        st.error("No sentiment analysis files found. Run FinBERT first.")
        return None

    df_list = []
    for file in files:
        df = pd.read_csv(file)

        # Rename 'content' to 'Text' if 'Text' is missing
        if "Text" not in df.columns and "content" in df.columns:
            df.rename(columns={"content": "Text"}, inplace=True)

        df_list.append(df)

    # Merge all files
    merged_df = pd.concat(df_list, ignore_index=True)

    return merged_df

# Streamlit UI
st.title("📊 Market Sentiment Analysis for Startups")

# Load dataset
df = load_all_sentiment_files()
if df is not None:
    st.success(f"✅ {len(df)} news articles loaded from multiple sentiment files!")

    if "Text" not in df.columns:
        st.error("❌ No 'Text' column found in the dataset. Check input files.")
        st.stop()

    # User input for startup sector
    sector = st.text_input("🔍 Enter your startup sector (e.g., AI, FinTech, E-commerce):").lower()

    if sector:
        # Filter relevant insights using the "Text" column
        filtered_df = df[df["Text"].str.lower().str.contains(rf"\b{re.escape(sector)}\b", na=False, regex=True)]

        if not filtered_df.empty:
            st.subheader(f"📌 Market Sentiment for **{sector.capitalize()}** Sector")

            # Sentiment distribution
            sentiment_counts = filtered_df["FinBERT_Sentiment"].value_counts()
            fig, ax = plt.subplots()
            sentiment_counts.plot(kind="pie", autopct="%1.1f%%", ax=ax, colors=["red", "blue", "green"])
            ax.set_ylabel("")
            ax.set_title("Sentiment Distribution")
            st.pyplot(fig)

            # Show sample insights
            st.subheader("📝 Sample Business Insights")
            st.write(filtered_df[["Text", "FinBERT_Sentiment"]].head(5))

        else:
            st.warning("⚠️ No relevant insights found for this sector. Try another.")
