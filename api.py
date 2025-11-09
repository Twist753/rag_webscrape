"""
FastAPI Backend for Assessment Recommendation System
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from rag_engine import AssessmentRAG
import uvicorn

# Initialize FastAPI app
app = FastAPI(title="SHL Assessment Recommendation API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG engine
rag_engine = AssessmentRAG()

# Load assessments on startup
@app.on_event("startup")
async def startup_event():
    print("Loading assessment data...")
    rag_engine.load_and_index_assessments()
    print("API ready!")

# Request/Response models
class RecommendRequest(BaseModel):
    query: str

class Assessment(BaseModel):
    name: str
    url: str

class RecommendResponse(BaseModel):
    recommendations: List[Assessment]

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Recommendation endpoint
@app.post("/recommend", response_model=RecommendResponse)
async def recommend_assessments(request: RecommendRequest):
    try:
        if not request.query or len(request.query.strip()) == 0:
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        # Get recommendations
        recommendations = rag_engine.recommend(request.query, top_k=10)
        
        # Ensure minimum 5 recommendations
        if len(recommendations) < 5:
            raise HTTPException(status_code=500, detail="Could not generate minimum 5 recommendations")
        
        # Format response
        formatted_recs = [
            Assessment(name=rec['name'], url=rec['url'])
            for rec in recommendations[:10]  # Max 10
        ]
        
        return RecommendResponse(recommendations=formatted_recs)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)