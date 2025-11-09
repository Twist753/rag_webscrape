import streamlit as st
import requests
import json

# --- Page Configuration ---
# This should be the first Streamlit command
st.set_page_config(
    page_title="SHL Assessment Recommender",
    layout="wide" # 1. Changed from "centered" to "wide"
)

# --- Configuration ---
# URL of your FastAPI backend.
API_URL = "http://127.0.0.1:8000/recommend" # IMPORTANT: Change this to your Render URL when deployed

# --- Custom CSS Styling ---
SHL_TEAL = "#00a99d"
SHL_LIGHT_GREY = "#f4f4f4"
OFF_WHITE = "#fafafa" # For background
LIGHT_GREEN_FADE = "#a3dcbb" # 1. Your new vibrant color

st.markdown(f"""
<style>
    /* --- 1. Background Gradient --- */
    /* Targets the main app container.
      This creates a fixed vertical gradient from off-white to a light green/teal.
      !important is used to override Streamlit's default theme (light/dark mode).
    */
    .stApp {{
        /* 1. Changed gradient to start 50% from the top */
        background-image: linear-gradient(to bottom, {OFF_WHITE} 50%, {LIGHT_GREEN_FADE} 100%) !important;
        background-attachment: fixed !important;
        background-size: cover !important;
        padding-bottom: 70px !important; /* 2. Added padding to bottom for fixed footer */
    }}

    /* --- 2. Header Styling --- */
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
        font-size: 1.6em; /* 2. Increased font size */
        font-weight: bold;
        color: #333;
    }}
    .header-links a {{
        margin-left: 15px;
        color: #555;
        text-decoration: none;
        font-size: 1.1em; /* 2. Increased font size */
    }}
    .header-links a:hover {{
        color: {SHL_TEAL};
    }}
    .header-links svg {{
        width: 28px; /* Slightly larger icons */
        height: 28px;
        fill: currentColor; /* Allows color to be set by parent 'a' tag */
    }}

    /* --- 3. Footer Styling --- */
    .footer {{
        width: 100%;
        text-align: center;
        padding: 20px 0 10px 0;
        font-size: 0.9em;
        color: #888;
        border-top: 1px solid #e0e0e0;
        
        /* 2. Added styles for fixed footer */
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background-color: {OFF_WHITE}; /* Give it a solid background */
        z-index: 100;
    }}

    /* --- Original Styles --- */
    /* Title color */
    h1 {{
        color: {SHL_TEAL};
    }}

    /* 2. Target st.subheader */
    h2 {{
        font-size: 1.75rem !important;
    }}

    /* Streamlit Button */
    .stButton > button {{
        background-color: {SHL_TEAL};
        color: #ffffff;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: bold;
        font-size: 1.05em; /* 2. Increased font size */
    }}
    .stButton > button:hover {{
        background-color: #007a70; /* A darker teal for hover */
        color: #ffffff;
    }}

    /* --- Custom Recommendation Card --- */
    .card {{
        background-color: {SHL_LIGHT_GREY};
        border-radius: 10px;
        padding: 18px;
        margin-bottom: 12px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border: 1px solid #e0e0e0;
    }}
    .card-title {{
        font-size: 1.25em; /* 2. Increased font size */
        font-weight: bold;
        color: #333333;
        margin-bottom: 5px;
    }}
    .card-details {{
        font-size: 1.05em; /* 2. Increased font size */
        color: #555555;
        margin-bottom: 15px;
    }}
    .card-link a {{
        color: {SHL_TEAL};
        text-decoration: none;
        font-weight: bold;
        font-size: 1.05em; /* 2. Increased font size */
    }}
    .card-link a:hover {{
        text-decoration: underline;
    }}
</style>
""", unsafe_allow_html=True)

# --- 2. Header ---
# Using columns is tricky for full-width elements, so st.markdown is better.
# SVGs for LinkedIn and GitHub are inlined for portability.
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

# Use a text_area for longer inputs like Job Descriptions
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

# --- 3. Footer ---
st.markdown("""
<div class="footer">
    SHL Assessment Recommender | Built with coffee ;)
</div>
""", unsafe_allow_html=True)