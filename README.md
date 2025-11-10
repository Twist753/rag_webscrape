# SHL Assessment Recommendation System

A GenAI-based recommendation system that helps recruiters find relevant SHL assessments using natural language queries. Built with an advanced Retrieval-Augmented Generation (RAG) pipeline for accurate, balanced recommendations.

[![Live Demo](https://img.shields.io/badge/Live-Demo-brightgreen)](https://shl-project-abhinavtyagi-2103.streamlit.app/)
[![API](https://img.shields.io/badge/API-Endpoint-blue)](https://rag-webscrape.onrender.com)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)

## Live Application

- **Frontend**: [https://shl-project-abhinavtyagi-2103.streamlit.app/](https://shl-project-abhinavtyagi-2103.streamlit.app/)
- **API Endpoint**: `https://rag-webscrape.onrender.com/recommend`

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Technical Stack](#technical-stack)
- [Project Structure](#project-structure)
- [API Documentation](#api-documentation)
- [Installation](#installation)
- [Usage](#usage)
- [Performance](#performance)
- [Contributing](#contributing)
- [Contact](#contact)

## Overview

This system addresses the inefficiency of manual, keyword-based assessment searches by implementing a sophisticated RAG pipeline that:
- Performs semantic search using vector embeddings
- Applies structured metadata filtering
- Leverages LLM-driven query understanding
- Delivers balanced recommendations from 310+ SHL assessments

**Performance**: Achieved **Mean Recall@10 ≈ 0.5** on training data

## Features

- **Semantic Search**: Natural language understanding for job descriptions
- **Multi-criteria Filtering**: Duration, test type, skills, and job level matching
- **Balanced Recommendations**: Round-robin algorithm ensures diverse test type coverage
- **LLM-Enhanced Parsing**: Google Gemini API for intelligent query interpretation
- **Robust Fallback**: Heuristic parser ensures system reliability
- **Real-time Results**: Fast retrieval from ChromaDB vector store

## Architecture

### RAG Pipeline
```
User Query
    ↓
[LLM Query Parser] ← Google Gemini API
    ↓
{skills, test_types, duration, job_levels}
    ↓
[ChromaDB Retrieval] ← Hybrid Search (Vector + Metadata)
    ↓
Top 60 Candidates
    ↓
[Re-ranking & Balancing] ← Job Level Boost + Round-Robin
    ↓
Top 5-10 Recommendations
```

### Core Components

#### 1. Data Indexing (`embeddings.py`)

**Metadata Extraction**:
- `duration_mins`: Intelligent parsing ("1 hour" → 60, "30-40 min" → 35)
- `test_type`: Assessment category (K, P, C)
- `skills`: Regex-based keyword extraction with whole-word matching
- `job_levels`: Parsed from descriptions

**Document Creation**:
- Natural-language document combining name, description, skills, and job levels
- Embedded using `sentence-transformers/all-MiniLM-L6-v2`
- Stored in ChromaDB with hybrid vector + metadata storage

#### 2. Recommendation Logic (`rag_engine.py`)

**Multi-Stage Pipeline**:

1. **Query Parsing**: 
   - LLM-based structured extraction (Gemini 2.5 Flash)
   - Fallback heuristic parser for reliability

2. **Filtered Retrieval**:
   - ChromaDB `where` filter for duration and test type
   - Semantic vector search on enhanced query
   - Retrieves top 60 candidates

3. **Re-ranking**:
   - Job level matching boost (1.1x multiplier)
   - Relevance score optimization

4. **Balancing**:
   - Round-robin algorithm for multi-type queries
   - Ensures diverse recommendations across K, P, C categories

## 🛠️ Technical Stack

| Component | Technology |
|-----------|------------|
| Backend API | FastAPI, Uvicorn |
| Frontend | Streamlit |
| Vector Database | ChromaDB |
| Embedding Model | sentence-transformers/all-MiniLM-L6-v2 |
| LLM | Google Gemini API (gemini-2.5-flash) |
| Web Scraping | BeautifulSoup4, Requests |
| Deployment | Render (Backend), Streamlit Cloud (Frontend) |

## Project Structure
```
rag_webscrape/
├── scraper.py                 # SHL catalog scraper
├── embeddings.py              # Vector database builder
├── rag_engine.py              # Core RAG logic
├── backend.py                 # FastAPI application
├── app.py                     # Streamlit frontend
├── data/
│   ├── assessments.json       # Scraped assessment data (310 items)
│   ├── skills_data.py         # Tech/soft skills lists
│   ├── train.csv              # Training data for evaluation
│   └── test.csv               # Test data for predictions
├── submission/
│   └── abhinav_tyagi.csv      # Final predictions
├── requirements.txt           # Python dependencies
├── render.yaml                # Render deployment config
├── .env.example               # Environment variables template
└── README.md                  # This file
```

## 📡 API Documentation

### Base URL
```
https://rag-webscrape.onrender.com
```

### Endpoints

#### 1. Health Check
```http
GET /health
```

**Response**:
```json
{
  "status": "healthy"
}
```

#### 2. Get Recommendations
```http
POST /recommend
```

**Request Body**:
```json
{
  "query": "I am hiring for Java developers who can collaborate with business teams. Need 40 minute assessment."
}
```

**Response**:
```json
{
  "recommendations": [
    {
      "name": "Java Programming Test",
      "url": "https://www.shl.com/solutions/products/...",
      "duration": "40",
      "test_type": "K"
    },
    {
      "name": "Business Communication Assessment",
      "url": "https://www.shl.com/solutions/products/...",
      "duration": "35",
      "test_type": "C"
    }
  ]
}
```

### cURL Example
```bash
curl -X POST https://rag-webscrape.onrender.com/recommend \
  -H "Content-Type: application/json" \
  -d '{"query": "Marketing Manager with 5 years experience, 30 min test"}'
```

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Google Gemini API key

### Setup Steps

1. **Clone the repository**
```bash
git clone https://github.com/Twist753/rag_webscrape.git
cd rag_webscrape
```

2. **Create virtual environment**
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**
```bash
# Create .env file
cp .env.example .env

# Add your Gemini API key
echo 'GEMINI_API_KEY="your_api_key_here"' > .env
```

5. **Build the vector database**
```bash
python embeddings.py
```
This will process `data/assessments.json` and create the ChromaDB index.

## Usage

### Running Locally

#### Start the Backend API
```bash
uvicorn backend:app --host 0.0.0.0 --port 8000 --reload
```
API will be available at `http://localhost:8000`

#### Start the Frontend
Open a new terminal and run:
```bash
streamlit run app.py
```
Frontend will open at `http://localhost:8501`

### Using the Web Interface

1. Navigate to the Streamlit app
2. Enter a job description or query (e.g., "Java developer with business skills, 40 min test")
3. Click "Get Recommendations"
4. View balanced recommendations with assessment details
5. Click "Add Example" to cycle through sample queries

### Using the API Directly

**Python Example**:
```python
import requests

url = "https://rag-webscrape.onrender.com/recommend"
payload = {
    "query": "Looking for Python developer assessment under 30 minutes"
}

response = requests.post(url, json=payload)
recommendations = response.json()["recommendations"]

for rec in recommendations:
    print(f"{rec['name']} - {rec['duration']} mins - {rec['url']}")
```

**JavaScript Example**:
```javascript
fetch('https://rag-webscrape.onrender.com/recommend', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    query: 'Marketing manager with event management skills'
  })
})
.then(res => res.json())
.then(data => console.log(data.recommendations));
```

## Performance

### Evaluation Metrics
- **Mean Recall@10**: ~0.5 on training data
- **Hybrid Search**: Vector similarity + metadata filtering
- **Balanced Output**: Round-robin ensures diverse recommendations

### Key Improvements Over Baseline
- Multi-stage RAG pipeline vs. naive embedding
- LLM-enhanced query understanding
- Metadata-aware filtering (duration, type, skills)
- Job level boosting for relevance
- Balanced recommendation across test types

### Webpage view

<img width="1919" height="865" alt="Screenshot 2025-11-10 111607" src="https://github.com/user-attachments/assets/d80ff3fe-5753-475e-b4f7-59f19bb4a71f" />

## Contact

**Abhinav Tyagi**
- Email: abhinavty753@gmail.com
- LinkedIn: [abhinav-tyagi-73373b281](https://www.linkedin.com/in/abhinav-tyagi-73373b281/)
- GitHub: [@Twist753](https://github.com/Twist753)

## License

This project is part of an assessment submission for SHL.

---

**Built with lots of coffee by Abhinav Tyagi :)**

