import streamlit as st

def show():
    st.sidebar.title("Navigation")
    return st.sidebar.radio("Go to", 
                            ["Landing Page", "Market Sentiment", "Valuation Prediction", 
                             "Funding Prediction", "Investment Recommendations", "Chatbot", "Pitch Generator"])
