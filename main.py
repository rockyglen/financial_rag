# main.py

from fastapi import FastAPI
from pydantic import BaseModel
from rag_chain import rag_chain
from typing import List
from fastapi.middleware.cors import CORSMiddleware
import os


# --- 1. Pydantic Schemas ---
class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    source_documents: List[str]


# --- 2. FastAPI Setup ---
app = FastAPI(
    title="Financial RAG Assistant Microservice",
    description="A production-ready RAG system deployed on AWS EC2.",
)

# Enable CORS for the Streamlit frontend
# IMPORTANT: When deploying Streamlit, change "http://localhost:8501"
# to your Streamlit Cloud URL (e.g., https://your-app.streamlit.app)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- 3. The Prediction Endpoint ---
@app.post("/predict", response_model=QueryResponse)
async def predict_answer(request: QueryRequest):
    """
    Endpoint to answer questions using the RAG chain, returning the answer and source documents.
    """
    print(f"\n[API Request] Received question: {request.question}")
    try:
        result = rag_chain.invoke({"question": request.question})

        return result

    except Exception as e:
        print(f"[API Error] An error occurred: {e}")
        return QueryResponse(
            answer="An internal server error occurred while processing the request. Please check the server logs.",
            source_documents=[],
        )
