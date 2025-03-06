import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from components import valuation, sentiment, investment, funding, chatbot,pitch_generator  # Add chatbot import

# Set page config
st.set_page_config(
    page_title="PitchGen AI",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    @import url('https://fonts.googleapis.com/icon?family=Material+Icons');
    @import url('https://fonts.googleapis.com/css2?family=Lora:wght@400;500;600;700&display=swap');
    /* Main Elements */
    body {
        font-family: 'Times New Roman', serif;
        background-color: #f8f9fa;
        font-size: 21px; /* Increase font size */
    }

    /* Sidebar Styling */
    .css-1d391kg, .css-1lcbmhc, [data-testid="stSidebar"] {
        background-color: #1b096b !important;
    }

    /* Increase sidebar text size */
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        font-size: 21px !important;
    }

    .stButton > button {
        background-color: #1b096b;
        color: white;
        border-radius: 4px;
        border: none;
        padding: 12px 24px;
        font-weight: 500;
        font-size: 25px !important;  /* Increase button text size */
        width: 100%;
        text-align: center;
        margin: 4px 0;
    }

    .stButton > button:hover {
        background-color: #1e0880;
        color:white;
        border: 1px solid white;
    }

    /* Standardize button hover color */
    .stButton > button:hover {
        background-color: #1e0880 !important;
        color: white !important;
        border: 1px solid white !important;
    }

    /* Logo area in sidebar */
    .logo-container {
        background-color: #1b096b;
        padding: 15px 20px;
        border-radius: 8px;
        margin-bottom: 30px;
        color: white;
        font-size: 32px !important;  /* Increase logo text size */
        font-weight: 700;
        text-align: center;
    }

    /* Cards */
    .card {
        border-radius: 8px;
        background-color: white;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        border: 1px solid #e5e7eb;
    }

    .card-title {
        font-size: 20px; /* Increase card title font size */
        font-weight: 600;
        color: #1e3a8a;
        margin-bottom: 15px;
    }

    /* Form elements */
    .stTextInput > div > div > input {
        border-radius: 4px;
        background-color: #f1f5f9;
    }

    .stSelectbox > div > div {
        border-radius: 4px;
        background-color: #f1f5f9;
    }

    .stButton > button {
        background-color: #1b096b; /* Change button color */
        color: white;
        border-radius: 4px;
        border: none;
        padding: 12px 24px; /* Increase padding */
        font-weight: 500;
        font-size: 30px; /* Increase button font size */
        display: block; /* Make buttons full width */   display: block; /* Make buttons full width */
        width: 100%; /* Make buttons full width */        margin: 0 auto; /* Center the buttons */
        text-align: center;
        opacity: 1 !important; /* Center align text */ Center align text */
    }

    .stButton > button:hover {    .stButton > button:hover {
        background-color: #1e0880;lor: white;
    }
                /* Active button state */
    .stButton > button:active, 
    .stButton > button[data-active="true"] {
        background-color: white !important;
        color: white !important;
        border: 1px solid white !important;
        opacity: 1 !important;
    }


    /* Header area */
    .header {
        padding: 20px;
        margin-bottom: 20px;
        border-radius: 8px;
        background-color: white;
        border: 1px solid #e5e7eb;
    }

    .header h1 {
        color: #1e3a8a;
        font-weight: 700;
        font-size: 32px; /* Increase header font size */
        margin-bottom: 5px;
    }

    .header p {
        color: #6b7280;
        font-size: 20px; /* Increase header paragraph font size */
    }

    /* Navigation item */
    .nav-item {
        display: flex;
        align-items: center;
        padding: 10px 15px;
        margin-bottom: 5px;
        border-radius: 4px;
        color: white;
        text-decoration: none;
    }

    .nav-item.active {
        background-color: #2563eb;
    }

    .nav-item:hover:not(.active) {
        background-color: rgba(255,255,255,0.1);
    }

    .nav-icon {
        background-color: rgba(255,255,255,0.1);
        width: 30px;
        height: 30px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-right: 15px;
    }

    .nav-text {
        font-size: 18px; /* Increase nav text font size */
        font-weight: 400;
    }

    .nav-item.active .nav-text {
        font-weight: 500;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #6b7280;
        font-size: 14px; /* Increase footer font size */
        margin-top: 20px;
        padding: 10px;
    }

    /* User profile */
    .user-profile {
        display: flex;
        align-items: center;
        margin-top: 30px;
        padding: 15px;
    }

    .user-avatar {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        background-color: #bfdbfe;
        color: #1e3a8a;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 600;
        margin-right: 15px;
    }

    .user-name {
        color: white;
        font-size: 25px !important;  /* Increase user name size */
        font-weight: 500;
    }

    /* Chat UI */
    .chat-container {
        background-color: #f1f5f9;
        border-radius: 4px;
        padding: 10px;
        height: 120px;
        overflow-y: auto;
        margin-bottom: 10px;
    }

    .chat-bubble-user {
        background-color: #dbeafe;
        color: #1e3a8a;
        border-radius: 15px;
        padding: 10px 16px; /* Increase padding */
        display: inline-block;
        max-width: 80%;
        margin-bottom: 10px;
        font-size: 18px; /* Increase chat bubble font size */
    }

    .chat-bubble-ai {
        background-color: #1e3a8a;
        color: white;
        border-radius: 15px;
        padding: 10px 16px; /* Increase padding */
        display: inline-block;
        max-width: 80%;
        margin-bottom: 10px;
        margin-left: 20%;
        font-size: 18px; /* Increase chat bubble font size */
    }

    /* View details link */
    .view-details {
        text-align: right;
        color: #3b82f6;
        font-weight: 500;
        font-size: 16px; /* Increase view details font size */
        margin-top: 10px;
    }

    /* Logo area */
    .logo-container {
        background-color: #1b096b;
        padding: 15px 20px;
        border-radius: 8px;
        margin-bottom: 30px;
        color: white;
        font-size: 28px; /* Increase logo font size */
        font-weight: 700;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for navigation
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'home'

# Create the sidebar
with st.sidebar:
    # Logo
    st.markdown('<div class="logo-container">PitchGen AI 🚀</div>', unsafe_allow_html=True)
    # Navigation buttons
    if st.button("🏠 Landing Page"):
        st.session_state.current_page = 'home'
    
    if st.button("💰 Valuation Prediction"):
        st.session_state.current_page = 'valuation'
    
    if st.button("📈 Market Sentiment"):
        st.session_state.current_page = 'sentiment'
    
    if st.button("📊 Funding Prediction"):
        st.session_state.current_page = 'funding'

    if st.button("📊 Investor Lookup"):
        st.session_state.current_page = 'investment'
    
    if st.button("🚀 Pitch Generator"):
        st.session_state.current_page = 'pitch_generator'
    
    # User profile at bottom of sidebar
    st.markdown("""
    <div style="position: fixed; bottom: 0; width: 17%; padding-bottom: 30px;">
        <div class="user-profile">
            <div class="user-avatar">NMBD</div>
            <div class="user-name">AI Craft</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Main content conditional rendering
if st.session_state.current_page == 'valuation':
    valuation.show()
elif st.session_state.current_page == 'sentiment':
    sentiment.show()
elif st.session_state.current_page == 'funding':
    funding.show()
elif st.session_state.current_page == 'investment':
    investment.show()
elif st.session_state.current_page == 'pitch_generator':
    pitch_generator.show()

else:
    # Original home page content
    st.markdown("""
    <div class="header">
        <h1>Welcome to PitchGen AI</h1>
        <p>Your AI-powered platform for startup success</p>
    </div>
    """, unsafe_allow_html=True)

    # First row of widgets
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Market Sentiment</div>', unsafe_allow_html=True)
        
        # Generate sample data for line chart
        dates = pd.date_range(start='2024-01-01', periods=10, freq='M')
        values = [70, 72, 71, 75, 82, 79, 85, 82, 88, 86]
        df = pd.DataFrame({'date': dates, 'sentiment': values})
        
        # Create a line chart
        fig = px.line(df, x='date', y='sentiment', 
                    labels={'sentiment': 'Sentiment Score', 'date': 'Month'},
                    line_shape='spline')
        fig.update_traces(line_color='#3b82f6', line_width=3)
        fig.update_layout(
            margin=dict(l=20, r=20, t=10, b=30),
            height=160,
            plot_bgcolor='#f1f5f9',
            paper_bgcolor='white',
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='#cbd5e1')
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown('<div class="view-details">View Details →</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Valuation Trends</div>', unsafe_allow_html=True)
        
        # Sample data for bar chart
        categories = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']
        values = [20, 40, 60, 80, 100]
        
        # Create bar chart
        fig = px.bar(x=categories, y=values,
                    labels={'x': 'Quarter', 'y': 'Valuation ($M)'},
                    color_discrete_sequence=['#3b82f6'])
        fig.update_layout(
            margin=dict(l=20, r=20, t=10, b=30),
            height=160,
            plot_bgcolor='#f1f5f9',
            paper_bgcolor='white'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown('<div class="view-details">View Details →</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Funding Success</div>', unsafe_allow_html=True)
        
        # Donut chart for funding success
        labels = ['Success', 'Partial', 'Failed']
        values = [72, 18, 10]
        colors = ['#3b82f6', '#bfdbfe', '#dbeafe']
        
        fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.6, marker_colors=colors)])
        fig.update_layout(
            margin=dict(l=20, r=20, t=10, b=10),
            height=160,
            showlegend=False,
            annotations=[dict(text="72%", x=0.5, y=0.5, font_size=20, font_color='#1e3a8a', showarrow=False),
                        dict(text="Success Rate", x=0.5, y=0.4, font_size=12, font_color='#6b7280', showarrow=False)]
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown('<div class="view-details">View Details →</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Second row of widgets
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Generate Your Pitch</div>', unsafe_allow_html=True)
        name = st.text_input("Startup Name", placeholder="Enter your startup name...")
        area = st.text_input("Industry/Area", placeholder="Enter industry...")
        year_founded = st.number_input("Year Founded", min_value=1900, max_value=2100, value=2020, step=1)
        pitch_idea = st.text_area("Describe your startup idea in a concise sentence", placeholder="Enter your startup idea...")
        
        if st.button("Generate Pitch", key="generate_pitch"):
            with st.spinner("🚀 Generating your pitch..."):
                pitch_prompt = f"Generate a pitch for {name}, founded in {year_founded} in the {area} industry. The core idea is: {pitch_idea}"
                pitch_response = chatbot.get_ai_response(pitch_prompt)
                if pitch_response:
                    st.markdown(f'''
                    <div style="background-color: #dbeafe; padding: 20px; border-radius: 8px; margin-top: 20px;">
                        <h3 style="color: #1e3a8a; margin-bottom: 10px;">Your Generated Pitch</h3>
                        <p style="color: #1e3a8a;">{pitch_response}</p>
                    </div>
                    ''', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">🤖 AI Assistant</div>', unsafe_allow_html=True)
        
        # Initialize chat history for the assistant
        if "home_assistant_messages" not in st.session_state:
            st.session_state.home_assistant_messages = []

        # Display chat messages
        chat_container = st.container()        
        with chat_container:
            for message in st.session_state.home_assistant_messages[-3:]:  # Show only last 3 messages
                role = "User" if message['role'] == "user" else "AI"
                st.markdown(f"**{role}:** {message['content']}")

        # Chat input with unique key
        if "assistant_input_value" not in st.session_state:
            st.session_state.assistant_input_value = ""

        prompt = st.text_input("Ask me anything...", placeholder="Type here...", key="home_assistant_input", value=st.session_state.assistant_input_value)
        
        # **Send Button with Unique Key**
        if st.button("Send", key="send_home_assistant_btn") and prompt.strip():
            # Append user message
            st.session_state.home_assistant_messages.append({"role": "user", "content": prompt})

            # Get AI response using chatbot
            with st.spinner("🤖 Thinking..."):
                ai_response = chatbot.get_ai_response(prompt)
                if ai_response:
                    st.session_state.home_assistant_messages.append({"role": "ai", "content": ai_response})

            # **Clear the input field by updating session state**
            st.session_state.assistant_input_value = ""
            st.rerun()  # Refresh the UI to reset the input field

        # **Fix: Clear Chat Button with Unique Key**
        if st.button("Clear Chat", key="clear_home_assistant_btn"):
            st.session_state.home_assistant_messages = []  # Reset messages
            st.rerun()