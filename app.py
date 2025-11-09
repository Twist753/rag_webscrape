import streamlit as st
import requests
import pandas as pd

# -----------------------------------------------------
# BACKEND CONFIGURATION
# -----------------------------------------------------
# Change this to your deployed Render URL when deployed
BACKEND_URL = "http://127.0.0.1:8000"   # For local testing
# Example for Render: "https://shl-rag-backend.onrender.com"

# -----------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------
st.set_page_config(page_title="SHL Assessment Recommendation System", layout="wide")

st.title(" SHL Assessment Recommendation System")
st.markdown("""
Enter a **job description** or **hiring query**, and get up to 10 most relevant SHL assessments.
""")

# -----------------------------------------------------
# INPUT AREA
# -----------------------------------------------------
query = st.text_area("Job Description / Query:", height=150, placeholder="e.g. Hiring a Python developer with strong teamwork and analytical reasoning skills")

col1, col2 = st.columns([1, 5])
with col1:
    search_btn = st.button("🔍 Recommend")

# -----------------------------------------------------
# QUERY BACKEND
# -----------------------------------------------------
if search_btn:
    if not query.strip():
        st.warning("Please enter a valid query.")
    else:
        try:
            with st.spinner("Fetching recommendations..."):
                response = requests.post(f"{BACKEND_URL}/recommend", json={"query": query}, timeout=90)

            if response.status_code == 200:
                data = response.json()
                recs = data.get("recommendations", [])

                if len(recs) == 0:
                    st.warning("No recommendations found. Try a different query.")
                else:
                    st.success(f"Found {len(recs)} relevant assessments")

                    # Format results into DataFrame
                    df = pd.DataFrame(recs)

                    # Make URLs clickable
                    def make_link(row):
                        return f"[{row['name']}]({row['url']})"
                    df["Assessment"] = df.apply(make_link, axis=1)

                    # Final display
                    st.markdown("### Recommendations")
                    st.write(df[["Assessment"]].to_markdown(index=False), unsafe_allow_html=True)

            else:
                st.error(f"Error {response.status_code}: {response.text}")

        except requests.exceptions.RequestException as e:
            st.error(f"Could not connect to backend: {e}")

# -----------------------------------------------------
# HEALTH CHECK (optional small badge)
# -----------------------------------------------------
try:
    health = requests.get(f"{BACKEND_URL}/health", timeout=5)
    if health.status_code == 200:
        st.sidebar.success("Backend connected ... :)")
    else:
        st.sidebar.error(" Backend not responding")
except:
    st.sidebar.error(" Backend not reachable")
