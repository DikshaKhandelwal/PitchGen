import streamlit as st

st.set_page_config(page_title="PitchGen AI", layout="wide")

from components import sidebar, sentiment, valuation, funding, chatbot, investment, landing, pitch_generator

# Sidebar for navigation
page = sidebar.show()

if page == "Landing Page":
    landing.show()
elif page == "Market Sentiment":
    sentiment.show()
elif page == "Valuation Prediction":
    valuation.show()
elif page == "Funding Prediction":
    funding.show()
elif page == "Investment Recommendations":
    investment.show()
elif page == "Chatbot":
    chatbot.show()
elif page == "Pitch Generator":
    pitch_generator.show()  # Pass components
