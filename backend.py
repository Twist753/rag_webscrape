"""
FastAPI Backend for SHL Assessment Recommendation System
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from rag_engine import AssessmentRAG
import uvicorn

# -----------------------------------------------------
# Initialize FastAPI
# -----------------------------------------------------
app = FastAPI(title="SHL Assessment Recommendation API")

# Allow cross-origin requests (for Streamlit frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------
# Initialize RAG Engine (loads Chroma + Gemini)
# -----------------------------------------------------
rag_engine = AssessmentRAG()

@app.on_event("startup")
async def startup_event():
    print("✅ Backend initialized successfully — Chroma collection connected.")
    # If you ever want to rebuild ChromaDB from JSON, you can add that here.


# -----------------------------------------------------
# Pydantic Request/Response Models
# -----------------------------------------------------
class RecommendRequest(BaseModel):
    query: str

class Assessment(BaseModel):
    name: str | None = None
    url: str | None = None
    test_type: str | None = None
    duration: str | None = None
    skills: str | None = None
    description: str | None = None
    score: float | None = None

class RecommendResponse(BaseModel):
    recommendations: List[Assessment]


# Health Check Endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}


# Recommendation Endpoint
@app.post("/recommend", response_model=RecommendResponse)
async def recommend_assessments(request: RecommendRequest):
    if not request.query or len(request.query.strip()) == 0:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        # Get recommendations (5–10 results, already handled inside RAG)
        recs = rag_engine.recommend(request.query, top_k=10)

        if not recs or len(recs) < 1:
            raise HTTPException(status_code=404, detail="No recommendations found")

        formatted = [
            Assessment(
                name=r.get("name", "Unnamed"),
                url=r.get("url", ""),
                test_type=r.get("test_type", "N/A"),
                duration=str(r.get("duration", "N/A")),
                skills=r.get("skills", ""),
                description=r.get("description", ""),
                score=r.get("score", 0)
            )
            for r in recs
        ]

        return RecommendResponse(recommendations=formatted)

    except Exception as e:
        print(" ⚠ Error during recommendation:", e)
        raise HTTPException(status_code=500, detail=str(e))


# -----------------------------------------------------
# Run server (for local testing)
# -----------------------------------------------------
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
