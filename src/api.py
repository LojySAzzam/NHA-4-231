"""
api.py
------
Milestone 3 — FastAPI REST API

Wraps the RAG chain (src/rag_chain.py) as a production-ready HTTP server.

Endpoints:
    GET  /health          → confirms the service is up and models are loaded
    POST /ask             → full RAG pipeline: retrieve + generate answer
    GET  /search          → retrieval only, no generation (raw top-k docs)

Run locally:
    uvicorn src.api:app --reload --port 8000

Then test in your browser:
    http://localhost:8000/docs     ← interactive Swagger UI (auto-generated)

Or with curl:
    curl -X POST http://localhost:8000/ask \
         -H "Content-Type: application/json" \
         -d '{"question": "I want to cancel my order"}'
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
import src.rag_chain as rag


# ── Request / Response schemas ────────────────────────────────────────────────

class AskRequest(BaseModel):
    message   : str  = Field(..., min_length=3, example="I want to cancel my order")
    top_k      : int  = Field(3, ge=1, le=10, description="Number of context docs to retrieve")
    use_hybrid : bool = Field(True, description="True = vector+BM25, False = pure vector")

class SourceDoc(BaseModel):
    score      : float
    category   : str
    intent     : str
    instruction: str
    response   : str
    title      : str = "Knowledge Base Document"  # Fallback string
    page       : int = 1                          # Fallback int

class AskResponse(BaseModel):
    query    : str
    answer   : str
    sources  : list[SourceDoc]
    retrieval: str

class SearchRequest(BaseModel):
    query      : str  = Field(..., min_length=3, example="cancel order")
    top_k      : int  = Field(5, ge=1, le=20)
    use_hybrid : bool = Field(True)

class HealthResponse(BaseModel):
    status : str
    models_loaded: bool


# ── App lifecycle ─────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all models once at startup, before accepting requests."""
    print("[API] Starting up — loading RAG models...")
    rag.load_all()
    print("[API] Ready to serve requests.")
    yield
    print("[API] Shutting down.")


# ── App definition ────────────────────────────────────────────────────────────

app = FastAPI(
    title="RAG Customer Support Chatbot API",
    description=(
        "Retrieval-Augmented Generation API for customer support automation. "
        "Built with FAISS vector search, sentence-transformers embeddings, "
        "and flan-t5-base local LLM. Milestone 3 of Project 5."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Allow all origins for local development
# Lock this down to specific domains before production deployment

origins = [
    "http://localhost:3000",  # If Next.js
    "http://localhost:5173",  # If Vite
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,     # Swapping "*" out for your specific frontend ports
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """
    Confirm the API is running and all models are loaded.
    Use this to verify the server started correctly.
    """
    return {
        "status"       : "ok",
        "models_loaded": rag._faiss_index is not None,
    }


@app.post("/api/chat", response_model=AskResponse, tags=["RAG"])
def ask_question(request: AskRequest):
    """
    Full RAG pipeline: retrieve relevant context, generate a natural answer.

    - Embeds your question using sentence-transformers
    - Retrieves the top-k most relevant support docs from the FAISS index
    - Feeds the docs as context into flan-t5-base
    - Returns the generated answer + the source documents it used
    """
    try:
        result = rag.ask(
            query=request.message,
            top_k=request.top_k,
            use_hybrid=request.use_hybrid,
        )
        # After you get 'result' from rag.ask(...)
        # ── Ensure unique titles/pages for React keys safely ──
        # This checks if result is a dictionary or an object
        sources = result.get("sources", []) if isinstance(result, dict) else getattr(result, "sources", [])

        # Ensure each source has a unique title/id for the React keys
        for i, source in enumerate(result.get("sources", [])):
            if isinstance(source, dict):
                # If sources are dictionaries
                category_name = source.get("category", "Knowledge Base Document")
                source["title"] = f"{category_name} (Source {i+1})"
                source["page"] = i + 1
            else:
                # If sources are objects
                category_name = getattr(source, "category", "Knowledge Base Document")
                source.title = f"{category_name} (Source {i+1})"
                source.page = i + 1
   
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/search", response_model=list[SourceDoc], tags=["RAG"])
def search_docs(message: str, top_k: int = 5, use_hybrid: bool = True):
    """
    Retrieval only — returns top-k matching support documents with no generation.
    Useful for debugging retrieval quality or building your own generation layer.
    """
    try:
        results = rag.search(query=message, top_k=top_k, use_hybrid=use_hybrid)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
