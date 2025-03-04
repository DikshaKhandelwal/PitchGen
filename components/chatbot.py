import os
import asyncio
import streamlit as st
from together import Together
from dotenv import load_dotenv

# Load API Key
load_dotenv()
api_key = os.getenv("TOGETHER_API_KEY")
client = Together(api_key=api_key)

def show():
    st.title("🤖 Chatbot")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    async def generate_detailed_pitch(user_input):
        prompt = f"""
        Create a *comprehensive, investor-ready pitch deck* for the business idea: *{user_input}*.
        The deck should be *detailed, structured, and professional*. It must include:  

        ## 1️⃣ Problem Statement  
        - What problem are you solving?  
        - Who is affected by this problem?  
        - How big is the problem (market pain points, data-driven insights)?  

        ## 2️⃣ Solution  
        - How does your product/service solve this problem?  
        - What makes it unique?  
        - Key features and value proposition.  

        ## 3️⃣ Market Opportunity  
        - Total Addressable Market (TAM), Serviceable Available Market (SAM), and Serviceable Obtainable Market (SOM).  
        - Industry trends and market growth potential.  

        ## 4️⃣ Business Model  
        - How do you plan to make money?  
        - Revenue model (subscription, one-time purchase, freemium, etc.).  
        - Pricing strategy.  

        ## 5️⃣ Competitive Advantage  
        - Who are your competitors?  
        - What sets you apart? (Technology, branding, partnerships, network effects, etc.).  
        - SWOT analysis (Strengths, Weaknesses, Opportunities, Threats).  

        ## 6️⃣ Revenue Streams  
        - Primary revenue sources.  
        - Potential future monetization strategies.  

        ## 7️⃣ Go-To-Market Strategy  
        - How will you acquire customers? (Sales channels, digital marketing, partnerships, etc.).  
        - Growth and scaling plans.  

        ## 8️⃣ Team  
        - Key team members (Co-founders, advisors, key hires).  
        - Experience and relevant expertise.  

        ## 9️⃣ Financial Projections  
        - Projected revenue, costs, and profitability.  
        - Breakeven point analysis.  
        - Funding requirements and expected ROI.  

        ## 🔟 Funding Requirements  
        - How much capital do you need?  
        - How will the funds be used? (Product development, marketing, hiring, etc.).  
        - Potential return for investors.  

        The response should be *professional, structured, and investor-friendly*.
        """
        
        response = client.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.choices[0].message.content

    if user_input := st.chat_input("Describe your business idea..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("🤖 Thinking... Generating your investor-ready pitch deck..."):
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                pitch_response = loop.run_until_complete(generate_detailed_pitch(user_input))
            
            st.markdown(pitch_response)

        st.session_state.messages.append({"role": "assistant", "content": pitch_response})