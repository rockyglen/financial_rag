# main.py

from fastapi import FastAPI
from pydantic import BaseModel
from rag_chain import rag_chain
from typing import List


# --- 1. Pydantic Schemas (Input/Output Validation) ---
class QueryRequest(BaseModel):
    """Defines the expected input structure for the API."""

    question: str


class QueryResponse(BaseModel):
    """Defines the output structure from the API."""

    answer: str
    source_documents: List[str]  # Now includes sources for debugging/citation


# --- 2. FastAPI Setup ---
app = FastAPI(
    title="Financial RAG Assistant Microservice",
    description="A production-ready RAG system using local LLMs on M4.",
)


# --- 3. The Prediction Endpoint ---
@app.post("/predict", response_model=QueryResponse)
async def predict_answer(request: QueryRequest):
    """
    Endpoint to answer questions using the RAG chain, returning the answer and source documents.
    """
    print(f"\n[API Request] Received question: {request.question}")
    try:
        # Invoke the RAG chain, which returns the structured dictionary
        result = rag_chain.invoke({"question": request.question})

        return result

    except Exception as e:
        print(f"[API Error] An error occurred: {e}")
        # Return a professional error message and empty source list
        return QueryResponse(
            answer="An internal server error occurred while processing the request. Please check the server logs.",
            source_documents=[],
        )


# --- EXECUTION ---
# To run the app: uvicorn main:app --reload
