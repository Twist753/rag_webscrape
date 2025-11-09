import streamlit as st
import requests
import json

# --- Page Configuration ---
# This should be the first Streamlit command
st.set_page_config(
    page_title="SHL Assessment Recommender",
    layout="centered"
)

# --- Configuration ---
# URL of your FastAPI backend.
# If running locally: "http://127.0.0.1:8000/recommend"
# If deployed on Render: "https://your-render-app-name.onrender.com/recommend"
API_URL = "http://127.0.0.1:8000/recommend" # IMPORTANT: Change this to your Render URL when deployed

# --- Custom CSS Styling ---
# This injects CSS to style the app like the SHL website
SHL_TEAL = "#00a99d" # Main brand color from SHL website
SHL_LIGHT_GREY = "#f4f4f4" # Light grey for card background

st.markdown(f"""
<style>
    /* Main App background */
    .stApp {{
        background_color: #ffffff; 
    }}

    /* Title color */
    h1 {{
        color: {SHL_TEAL};
    }}

    /* Streamlit Button */
    .stButton > button {{
        background-color: {SHL_TEAL};
        color: #ffffff;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: bold;
    }}
    .stButton > button:hover {{
        background-color: #007a70; /* A darker teal for hover */
        color: #ffffff;
    }}

    /* --- Custom Recommendation Card --- */
    .card {{
        background-color: {SHL_LIGHT_GREY};
        border-radius: 10px;          /* Rounded corners */
        padding: 18px;                /* Inner spacing */
        margin-bottom: 12px;          /* Space between cards */
        box-shadow: 0 2px 4px rgba(0,0,0,0.05); /* Subtle shadow */
        border: 1px solid #e0e0e0;
    }}
    .card-title {{
        font-size: 1.15em;
        font-weight: bold;
        color: #333333;
        margin-bottom: 5px;
    }}
    .card-details {{
        font-size: 0.95em;
        color: #555555;
        margin-bottom: 15px;
    }}
    .card-link a {{
        color: {SHL_TEAL};
        text-decoration: none;
        font-weight: bold;
    }}
    .card-link a:hover {{
        text-decoration: underline;
    }}
</style>
""", unsafe_allow_html=True)


# --- Frontend Application ---

# Display the SHL logo
st.image("https://www.shl.com/static/media/logo.759f2328.svg", width=100)

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
        with st.spinner("🧠 Analyzing your query and finding assessments..."):
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
                        st.subheader("Your Recommended Assessments")
                        
                        # Loop over recommendations and display in custom cards
                        for rec in recommendations:
                            name = rec.get("name", "No Name Provided")
                            url = rec.get("url", "#")
                            
                            # --- TODO: Update this when your backend sends more data ---
                            # These are placeholders until you update your backend.py
                            # to send this information from your RAG engine.
                            duration = rec.get("duration", "N/A") 
                            test_type = rec.get("test_type", "N/A")
                            # -----------------------------------------------------------

                            # HTML for the custom card
                            card_html = f"""
                            <div class="card">
                                <div class="card-title">{name}</div>
                                <div class="card-details">
                                    {duration} | Test Type = {test_type}
                                </div>
                                <div class="card-link">
                                    <a href="{url}" target="_blank">View Assessment Details</a>
                                </div>
                            </div>
                            """
                            st.markdown(card_html, unsafe_allow_html=True)
                
                elif response.status_code == 404:
                    st.info("No recommendations found for this query.")
                else:
                    # Show a more detailed error from the API
                    error_detail = response.json().get("detail", "Unknown error")
                    st.error(f"Error from API: {error_detail} (Code: {response.status_code})")
            
            except requests.exceptions.ConnectionError:
                st.error(
                    "❌ Connection Error: Could not connect to the backend API. "
                    f"Is it running at `{API_URL}`?"
                )
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")