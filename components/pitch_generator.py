import os
import streamlit as st
from together import Together
from dotenv import load_dotenv
from fpdf import FPDF
from components import funding, valuation, sentiment, investment

# Load API Key
load_dotenv()
api_key = os.getenv("TOGETHER_API_KEY")
client = Together(api_key=api_key)

# Function to generate pitch
def generate_pitch(name, area, year_founded, problem, solution, market_opportunity, business_model, 
                   competitive_advantage, revenue_streams, go_to_market, team, financial_projections, 
                   funding_requirements, sentiment, valuation, funding, investment):
    prompt = f"""
    Create a *comprehensive, investor-ready pitch deck* for the startup **{name}**, founded in **{year_founded}** in the **{area}** industry, based on the following details:

    **Market Sentiment:**  
    {sentiment if sentiment else "Not provided"}

    **Valuation Prediction:**  
    ${valuation:,.2f}

    **Funding Prediction:**  
    {funding:.2f} rounds

    **Investment Recommendations:**  
    {investment if investment else "Not provided"}

    The deck should be *detailed, structured, and professional*. It must include:

    ## 1️⃣ Problem Statement  
    {problem}

    ## 2️⃣ Solution  
    {solution}

    ## 3️⃣ Market Opportunity  
    {market_opportunity}

    ## 4️⃣ Business Model  
    {business_model}

    ## 5️⃣ Competitive Advantage  
    {competitive_advantage}

    ## 6️⃣ Revenue Streams  
    {revenue_streams}

    ## 7️⃣ Go-To-Market Strategy  
    {go_to_market}

    ## 8️⃣ Team  
    {team}

    ## 9️⃣ Financial Projections  
    {financial_projections}

    ## 🔟 Funding Requirements  
    {funding_requirements}

    The response should be *professional, structured, and investor-friendly*.
    """
    
    try:
        response = client.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ Error generating pitch: {e}"

# Function to generate a PDF pitch deck
def create_pdf(pitch_content):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", style="B", size=16)
    pdf.cell(200, 10, "Startup Pitch Deck", ln=True, align="C")
    
    pdf.set_font("Arial", size=12)
    pdf.ln(10)

    for line in pitch_content.split('\n'):
        pdf.multi_cell(0, 10, line.encode('latin-1', 'replace').decode('latin-1'))
        pdf.ln(2)  # Add spacing

    pdf_output = "pitch_deck.pdf"
    pdf.output(pdf_output)
    return pdf_output

# Function to fetch values from different pages
def fetch_component_output(component, use_component):
    """Safely fetch output from a component if the checkbox is selected."""
    if use_component:
        output = component.show()
        if isinstance(output, tuple):
            return output[1]  # Return only the output value
        return output
    return None

# Main Streamlit UI function
def show():
    st.title("🚀 Startup Pitch Generator")

    # Checkboxes for optional insights
    use_sentiment = st.checkbox("Include Market Sentiment Analysis", value=True)
    use_valuation = st.checkbox("Include Valuation Prediction", value=True)
    use_funding = st.checkbox("Include Funding Prediction", value=True)
    use_investment = st.checkbox("Include Investment Recommendations", value=True)

    # Fetch outputs dynamically before form submission
    sentiment_output = fetch_component_output(sentiment, use_sentiment) or "Not available"
    valuation_output = fetch_component_output(valuation, use_valuation) or 0.0
    funding_output = fetch_component_output(funding, use_funding) or 0.0
    investment_output = fetch_component_output(investment, use_investment) or "Not available"

    # Collect startup details from the user
    with st.form("pitch_form"):
        name = st.text_input("Startup Name")
        area = st.text_input("Industry/Area")
        year_founded = st.number_input("Year Founded", min_value=1900, max_value=2100, step=1)
        problem = st.text_area("Problem Statement")
        solution = st.text_area("Solution")
        market_opportunity = st.text_area("Market Opportunity")
        business_model = st.text_area("Business Model")
        competitive_advantage = st.text_area("Competitive Advantage")
        revenue_streams = st.text_area("Revenue Streams")
        go_to_market = st.text_area("Go-To-Market Strategy")
        team = st.text_area("Team")
        financial_projections = st.text_area("Financial Projections")
        funding_requirements = st.text_area("Funding Requirements")

        submit_button = st.form_submit_button("Generate Pitch")

    if submit_button:
        with st.spinner("Generating your detailed pitch..."):
            # Generate pitch content
            pitch_content = generate_pitch(
                name, area, year_founded, problem, solution, market_opportunity,
                business_model, competitive_advantage, revenue_streams, go_to_market, team,
                financial_projections, funding_requirements, sentiment_output, valuation_output, 
                funding_output, investment_output
            )

        # Display the generated pitch
        st.subheader("📜 Generated Pitch Deck Content")
        st.markdown(pitch_content)

        # Create and offer the pitch as a downloadable PDF
        pdf_path = create_pdf(pitch_content)
        with open(pdf_path, "rb") as pdf_file:
            st.download_button(label="📥 Download Pitch Deck PDF", data=pdf_file, file_name="pitch_deck.pdf", mime="application/pdf")

# Run the Streamlit app
if __name__ == "__main__":
    show()
