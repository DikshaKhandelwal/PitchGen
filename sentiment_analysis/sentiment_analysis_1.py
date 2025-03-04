import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import re

# Function to clean text
def clean_text(text):
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'[^A-Za-z0-9\s]+', '', text)  # Remove special characters
    return text.lower()

# Load dataset (new dataset: all-data.csv)
def load_dataset():
    df = pd.read_csv("all-data.csv", encoding='ISO-8859-1', header=None, names=['Sentiment', 'Text'])
  # Load dataset
    df['Cleaned_Text'] = df['Text'].apply(clean_text)
    sentiment_mapping = {"positive": 1, "neutral": 0, "negative": -1}  # Map sentiment labels
    df['Sentiment_Score'] = df['Sentiment'].map(sentiment_mapping)
    return df

# Streamlit UI
st.title("📊 Market Sentiment Analyzer for Startup Sectors")
st.write("Enter your startup sector to analyze market sentiment.")

# User input for startup sector
sector = st.text_input("Enter your startup sector (e.g., AI, FinTech, EdTech)")

if sector:
    df = load_dataset()
    filtered_df = df[df['Text'].str.contains(sector, case=False, na=False)]
    
    if not filtered_df.empty:
        avg_sentiment = filtered_df['Sentiment_Score'].mean()
        st.write(f"### Overall Market Sentiment for {sector}: ")
        
        if avg_sentiment > 0:
            st.success("The market sentiment is Positive ✅")
        elif avg_sentiment < 0:
            st.error("The market sentiment is Negative ❌")
        else:
            st.warning("The market sentiment is Neutral ⚖️")
        
        # Sentiment distribution
        st.write("### Sentiment Score Distribution")
        fig, ax = plt.subplots()
        filtered_df['Sentiment_Score'].hist(bins=30, ax=ax)
        ax.set_xlabel("Sentiment Score")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)
    else:
        st.warning("No data found for this sector. Try another keyword!")
