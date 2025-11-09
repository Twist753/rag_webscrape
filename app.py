import streamlit as st
import requests
import json

# --- Page Configuration ---
st.set_page_config(
    page_title="SHL Assessment Recommender",
    layout="wide" 
)

# --- Configuration ---
API_URL = "https://rag-webscrape.onrender.com/recommend" 

# --- Custom CSS Styling ---
SHL_TEAL = "#00a99d"
SHL_LIGHT_GREY = "#f4f4f4"
OFF_WHITE = "#fafafa" 
LIGHT_GREEN_FADE = "#a3dcbb" 

st.markdown(f"""
<style>
:root {{
    color-scheme: light !important;
}}
html, body, [class*="stAppViewContainer"], [class*="stApp"], [class*="main"], .stMarkdown, .stTextInput, .stTextArea {{
    background-color: {OFF_WHITE} !important;
    color: #000000 !important;
}}
/* Force labels and text inputs to use dark text on light background */
.stTextInput > div > div > input, 
.stTextArea > div > textarea {{
    background-color: #ffffff !important;
    color: #000000 !important;
    border: 1px solid #cccccc !important;
}}
/* Title & subheaders stay black 🔧 */
h1, h2, h3, label, .stMarkdown p {{
    color: #000000 !important;
}}
/* --- 1. Background Gradient --- */
.stApp {{
    background-image: linear-gradient(to bottom, {OFF_WHITE} 50%, {LIGHT_GREEN_FADE} 100%) !important;
    background-attachment: fixed !important;
    background-size: cover !important;
    padding-bottom: 70px !important;
}}
/* --- Header --- */
.header {{
    width: 100%;
    padding: 10px 0;
    margin-bottom: 20px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid #e0e0e0;
}}
.header-name {{
    font-size: 1.6em;
    font-weight: bold;
    color: #000000 !important;
}}
.header-links a {{
    margin-left: 15px;
    color: #333333;
    text-decoration: none;
    font-size: 1.1em;
}}
.header-links a:hover {{
    color: {SHL_TEAL};
}}
.header-links svg {{
    width: 28px;
    height: 28px;
    fill: currentColor;
}}
/* --- Footer --- */
.footer {{
    width: 100%;
    text-align: center;
    padding: 20px 0 10px 0;
    font-size: 0.9em;
    color: #444;
    border-top: 1px solid #e0e0e0;
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    background-color: {OFF_WHITE};
    z-index: 100;
}}
/* Buttons */
.stButton > button {{
    background-color: {SHL_TEAL};
    color: #ffffff;
    border: none;
    border-radius: 8px;
    padding: 10px 20px;
    font-weight: bold;
    font-size: 1.05em;
}}
.stButton > button:hover {{
    background-color: #007a70;
    color: #ffffff;
}}
/* Cards */
.card {{
    background-color: {SHL_LIGHT_GREY};
    border-radius: 10px;
    padding: 18px;
    margin-bottom: 12px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    border: 1px solid #e0e0e0;
}}
.card-title {{
    font-size: 1.25em;
    font-weight: bold;
    color: #000000 !important;
    margin-bottom: 5px;
}}
.card-details {{
    font-size: 1.05em;
    color: #333333 !important;
    margin-bottom: 15px;
}}
.card-link a {{
    color: {SHL_TEAL};
    text-decoration: none;
    font-weight: bold;
    font-size: 1.05em;
}}
.card-link a:hover {{
    text-decoration: underline;
}}
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown("""
<div class="header">
    <div class="header-name">Abhinav Tyagi - abhinavty753@gmail.com</div>
    <div class="header-links">
        <a href="https://www.linkedin.com/in/abhinav-tyagi-73373b281/" target="_blank">
            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24"><path d="M19 0h-14c-2.761 0-5 2.239-5 5v14c0 2.761 2.239 5 5 5h14c2.762 0 5-2.239 5-5v-14c0-2.761-2.238-5-5-5zm-11 19h-3v-11h3v11zm-1.5-12.268c-.966 0-1.75-.784-1.75-1.75s.784-1.75 1.75-1.75 1.75.784 1.75 1.75-.784 1.75-1.75 1.75zm13.5 12.268h-3v-5.604c0-3.368-4-3.113-4 0v5.604h-3v-11h3v1.765c1.396-2.586 7-2.777 7 2.476v6.759z"/></svg>
        </a>
        <a href="https://github.com/Twist753/rag_webscrape.git" target="_blank">
            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24"><path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.109-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/></svg>
        </a>
    </div>
</div>
""", unsafe_allow_html=True)


# --- Frontend Application ---
st.title("Assessment Recommendation System")
query = st.text_area(
    "Enter a job description or query:",
    placeholder="e.g., 'I am hiring for Java developers who can also collaborate effectively with my business teams.'"
)

if st.button("Get Recommendations"):
    if not query.strip():
        st.warning("Please enter a query or job description.")
    else:
        with st.spinner("Analyzing your query and finding assessments..."):
            try:
                # --- API Call ---
                payload = {"query": query}
                response = requests.post(API_URL, json=payload)

                if response.status_code == 200:
                    data = response.json()
                    recommendations = data.get("recommendations", [])

                    if not recommendations:
                        st.info("No recommendations found for this query.")
                    else:
                        st.subheader("Recommended Assessments")
                        
                        # Loop over recommendations and display in custom cards
                        for rec in recommendations:
                            name = rec.get("name", "Name not available")
                            url = rec.get("url", "URL not found")
                            duration = rec.get("duration", "N/A") 
                            test_type = rec.get("test_type", "N/A")

                            # HTML for the custom card
                            card_html = f"""
                            <div class="card">
                                <div class="card-title">{name}</div>
                                <div class="card-details">
                                    Test Duration(min) = {duration} | Test Type = {test_type}
                                </div>
                                <div class="card-link">
                                    <a href="{url}" target="_blank">Link for Assessment</a>
                                </div>
                            </div>
                            """
                            st.markdown(card_html, unsafe_allow_html=True)
                
                elif response.status_code == 404:
                    st.info("No recommendations found for this query.")
                else:
                    # Show a more detailed error from the API
                    error_detail = "Unknown error"
                    try:
                        error_detail = response.json().get("detail", "Unknown error")
                    except json.JSONDecodeError:
                        error_detail = response.text # Show raw text if not JSON
                    st.error(f"Error from API: {error_detail} (Code: {response.status_code})")
            
            except requests.exceptions.ConnectionError:
                st.error(
                    "Connection Error: Could not connect to the backend API. "
                    f"Is it running at `{API_URL}`?"
                )
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")

# --- Footer ---
st.markdown("""
<div class="footer">
    SHL Assessment Recommender | Built with coffee ;)
</div>
""", unsafe_allow_html=True)