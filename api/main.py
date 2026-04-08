"""FastAPI recommendation API."""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag.retriever import Retriever


app = FastAPI(title="Movie Recommender")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model once at startup
retriever = None

@app.on_event("startup")
def startup():
    global retriever
    retriever = Retriever()
    print("Retriever ready.")


class RecommendRequest(BaseModel):
    user_id: int
    k: int = 10


@app.post("/recommend")
def recommend(req: RecommendRequest):
    if req.k < 1 or req.k > 100:
        raise HTTPException(status_code=400, detail="k must be between 1 and 100")
    results = retriever.recommend(req.user_id, req.k)
    return {"recommendations": results}
