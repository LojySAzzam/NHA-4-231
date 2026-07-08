"""
api.py
------
Milestone 3 — FastAPI REST API (refactored)

Wraps the RAG chain (src/rag_chain.py) as a production-ready HTTP server.

Endpoints:
    GET  /health          -> confirms the service is up via rag.is_ready()
    POST /api/chat         }
    POST /ask              } same handler, two paths — see AskRequest below
    GET  /search          -> retrieval only, no generation (raw top-k docs)

Run locally:
    uvicorn src.api:app --reload --port 8000

Then test in your browser:
    http://localhost:8000/docs     <- interactive Swagger UI (auto-generated)

Or with curl (either payload shape works — see "API contract reconciliation"):
    curl -X POST http://localhost:8000/api/chat \
         -H "Content-Type: application/json" \
         -d '{"message": "I want to cancel my order"}'

    curl -X POST http://localhost:8000/ask \
         -H "Content-Type: application/json" \
         -d '{"question": "I want to cancel my order"}'

Refactor notes:
    - /health now calls the public rag.is_ready() instead of reaching into
      rag._faiss_index directly, restoring encapsulation across the module
      boundary.
    - AskRequest accepts either "message" (current frontend contract) or
      "question" (legacy documented contract) via a single field with
      AliasChoices, AND the handler is registered at both /api/chat and
      /ask, so neither consumer 404s or KeyErrors regardless of which
      path or payload shape they use.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, AliasChoices, ConfigDict
from contextlib import asynccontextmanager
import src.rag_chain as rag


# ── Request / Response schemas ────────────────────────────────────────────────

class AskRequest(BaseModel):
    """
    Accepts either payload shape without ambiguity:
        {"message": "..."}   <- current frontend (Next.js useChat-style) contract
        {"question": "..."}  <- legacy contract documented in the Milestone 3
                                 notebook / original module docstring
    Both populate the same `message` field via AliasChoices.
    """
    model_config = ConfigDict(populate_by_name=True)

    message: str = Field(
        ...,
        validation_alias=AliasChoices("message", "question"),
        min_length=3,
        examples=["I want to cancel my order"],
    )
    top_k      : int  = Field(3, ge=1, le=10, description="Number of context docs to retrieve")
    use_hybrid : bool = Field(True, description="True = RRF vector+BM25 fusion, False = pure vector")


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
        "Built with FAISS vector search + BM25 (fused via Reciprocal Rank "
        "Fusion), Gemini embeddings, and Groq (llama-3.3-70b-versatile) "
        "generation. Milestone 3 of Project 5."
    ),
    version="1.1.0",
    lifespan=lifespan,
)

# Restricted to known local frontend dev ports.
# Lock this down further to your actual deployed frontend domain before production.
origins = [
    "http://localhost:3000",  # Next.js
    "http://localhost:5173",  # Vite
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """
    Confirm the API is running and all runtime components are ready.

    Delegates entirely to rag.is_ready(), which checks the FAISS index,
    BM25 index, Groq client, and upstream API key presence from *inside*
    rag_chain.py — this endpoint no longer reaches into rag_chain's
    private module globals directly.
    """
    ready = rag.is_ready()
    return {
        "status"       : "ok" if ready else "degraded",
        "models_loaded": ready,
    }


def _ask_question_impl(request: AskRequest) -> AskResponse:
    """Shared implementation for both /api/chat and /ask."""
    try:
        result = rag.ask(
            query=request.message,
            top_k=request.top_k,
            use_hybrid=request.use_hybrid,
        )

        # Ensure each source has a unique title/page for stable React list keys.
        sources = result.get("sources", []) if isinstance(result, dict) else getattr(result, "sources", [])
        for i, source in enumerate(sources):
            if isinstance(source, dict):
                category_name = source.get("category", "Knowledge Base Document")
                source["title"] = f"{category_name} (Source {i + 1})"
                source["page"] = i + 1
            else:
                category_name = getattr(source, "category", "Knowledge Base Document")
                source.title = f"{category_name} (Source {i + 1})"
                source.page = i + 1

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat", response_model=AskResponse, tags=["RAG"])
def ask_question_chat(request: AskRequest):
    """
    Full RAG pipeline: retrieve relevant context (RRF hybrid fusion),
    generate a natural answer. Frontend contract: {"message": ...}.
    """
    return _ask_question_impl(request)


@app.post("/ask", response_model=AskResponse, tags=["RAG"], include_in_schema=False)
def ask_question_legacy(request: AskRequest):
    """
    Legacy-path alias for /api/chat, kept so the originally documented
    contract ({"question": ...} at POST /ask) never 404s. Hidden from the
    OpenAPI schema (include_in_schema=False) so /api/chat remains the one
    documented, canonical route going forward.
    """
    return _ask_question_impl(request)


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
