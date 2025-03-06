import os
import streamlit as st
from together import Together
from dotenv import load_dotenv

# Load API Key
load_dotenv()
api_key = os.getenv("TOGETHER_API_KEY")
together = Together(api_key=api_key)

def get_ai_response(prompt):
    try:
        response = together.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error getting response: {str(e)}")
        return None

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
    st.markdown("""
    <div style="background-color: white; padding: 20px; border-radius: 8px; border: 1px solid #e5e7eb; margin-bottom: 20px;">
        <h1 style="color: #1e3a8a; font-weight: 700; margin-bottom: 5px;">🤖 AI Assistant</h1>
        <p style="color: #6b7280;">Ask anything about your startup or get help with your pitch.</p>
    </div>
    """, unsafe_allow_html=True)

    # Initialize chat history
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    # Display chat messages
    for message in st.session_state.chat_messages:
        role = "User" if message['role'] == "user" else "AI"
        st.markdown(f"**{role}:** {message['content']}")

    # Chat input
    prompt = st.text_input("Ask me anything about your startup:", key="chat_input")

    if st.button("Send", key="send_message") and prompt:
        # Append user message
        st.session_state.chat_messages.append({"role": "user", "content": prompt})

        # Get AI response
        with st.spinner("🤖 Thinking..."):
            ai_response = get_ai_response(prompt)
            if ai_response:
                st.session_state.chat_messages.append({"role": "ai", "content": ai_response})

        # Reset text input
        st.session_state.chat_input = ""

    # Clear chat button
    if st.button("Clear Chat", key="clear_chat"):
        st.session_state.chat_messages = []

