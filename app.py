"""
Streamlit Frontend for SHL Assessment Recommender
"""

import streamlit as st
import requests
import pandas as pd

# Page config
st.set_page_config(
    page_title="SHL Assessment Recommender",
    page_icon="📋",
    layout="wide"
)

# Title
st.title("🎯 SHL Assessment Recommendation System")
st.markdown("Enter a job description or hiring query to get relevant assessment recommendations")

# API endpoint (change this when deployed)
API_URL = "http://localhost:8000"

# Check API health
try:
    health_response = requests.get(f"{API_URL}/health", timeout=5)
    if health_response.status_code == 200:
        st.sidebar.success("✅ API Connected")
    else:
        st.sidebar.error("❌ API Error")
except:
    st.sidebar.error("❌ API Not Reachable")
    st.info("Please make sure the API is running: `python api.py`")

# Input section
query = st.text_area(
    "Enter your query:",
    placeholder="Example: I am hiring for Java developers who can also collaborate effectively with my business teams.",
    height=100
)

# Recommend button
if st.button("🔍 Get Recommendations", type="primary"):
    if not query:
        st.warning("Please enter a query")
    else:
        with st.spinner("Finding best assessments..."):
            try:
                # Call API
                response = requests.post(
                    f"{API_URL}/recommend",
                    json={"query": query},
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    recommendations = data['recommendations']
                    
                    st.success(f"Found {len(recommendations)} relevant assessments")
                    
                    # Display as table
                    df = pd.DataFrame(recommendations)
                    df.index = df.index + 1
                    df.columns = ['Assessment Name', 'URL']
                    
                    # Make URLs clickable
                    df['URL'] = df['URL'].apply(lambda x: f'<a href="{x}" target="_blank">View Assessment</a>')
                    
                    st.markdown("### Recommended Assessments")
                    st.write(df.to_html(escape=False, index=True), unsafe_allow_html=True)
                    
                else:
                    st.error(f"API Error: {response.status_code}")
                    st.error(response.text)
                    
            except Exception as e:
                st.error(f"Error: {str(e)}")

# Sidebar with info
st.sidebar.markdown("### About")
st.sidebar.info(
    """
    This system uses RAG (Retrieval Augmented Generation) to recommend 
    relevant SHL assessments based on your hiring needs.
    
    **Features:**
    - Semantic search across 100+ assessments
    - Balanced recommendations for multi-domain queries
    - Returns 5-10 most relevant assessments
    """
)

st.sidebar.markdown("### Sample Queries")
st.sidebar.code("I am hiring for Java developers who can also collaborate effectively")
st.sidebar.code("Looking to hire mid-level professionals proficient in Python and SQL")
st.sidebar.code("Need cognitive and personality tests for analyst role")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using ChromaDB, Sentence Transformers, FastAPI & Streamlit")