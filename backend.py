"""
FastAPI Backend for SHL Assessment Recommendation System
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List
from rag_engine import AssessmentRAG
import uvicorn
import traceback

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
try:
    rag_engine = AssessmentRAG()
    print("✅ RAG Engine initialized successfully.")
except Exception as e:
    print("❌ Failed to initialize RAG Engine:", e)
    rag_engine = None


@app.on_event("startup")
async def startup_event():
    print("Backend initialized successfully — Chroma collection connected.")


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


# -----------------------------------------------------
# Health Check Endpoint
# -----------------------------------------------------
@app.get("/health")
async def health_check():
    return {"status": "healthy"}


# -----------------------------------------------------
# Root Endpoint (for Render ping)
# -----------------------------------------------------
@app.get("/")
async def root():
    return {"message": "SHL Assessment Recommendation Backend is running."}


# -----------------------------------------------------
# Recommendation Endpoint (Safe version)
# -----------------------------------------------------
@app.post("/recommend", response_model=RecommendResponse)
async def recommend_assessments(request: RecommendRequest):
    if not request.query or len(request.query.strip()) == 0:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    if rag_engine is None:
        return JSONResponse(
            status_code=503,
            content={
                "detail": "⚠ Recommendation service is initializing. Please try again in a few seconds."
            },
        )

    try:
        recs = rag_engine.recommend(request.query)

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
                score=r.get("score", 0),
            )
            for r in recs
        ]

        return RecommendResponse(recommendations=formatted)

    except TimeoutError:
        print("⚠ Timeout: RAG pipeline took too long.")
        return JSONResponse(
            status_code=504,
            content={
                "detail": "⚠ The recommendation service is taking longer than expected. Please retry after a few seconds."
            },
        )

    except ConnectionError:
        print("⚠ Connection issue with RAG engine.")
        return JSONResponse(
            status_code=502,
            content={
                "detail": "⚠ The recommendation service is temporarily unreachable. Please try again soon."
            },
        )

    except Exception as e:
        print("⚠ Error during recommendation:", e)
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={
                "detail": "The recommendation service encountered cold start issues(due to free tier deployment). Please try again later."
            },
        )

# -----------------------------------------------------
# Run server (for local testing)
# -----------------------------------------------------
if __name__ == "__main__":
    uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=True)
