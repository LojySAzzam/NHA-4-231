"""
rag_chain.py
------------
Milestone 3 — Core RAG logic (Gemini 2.0 Flash + text-embedding-004)

Pipeline:
    user query
        -> embed query (Google text-embedding-004, 768-dim)
        -> retrieve top-k context docs (FAISS)
        -> build prompt with context
        -> generate answer (Gemini 2.0 Flash via Google AI Studio)
        -> return answer + sources

Swap guide (when Azure is ready):
    - Replace _retrieve_hybrid() with Azure SearchClient call
    - Everything else stays identical
"""

import os
import re
import pickle
import numpy as np
import pandas as pd
import faiss
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
import google.generativeai as genai

# ── Load environment variables from .env ──────────────────────────────────────
load_dotenv()

GEMINI_API_KEY       = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL_NAME    = os.getenv("GEMINI_MODEL_NAME",    "gemini-2.0-flash")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "gemini-embedding-001")
EMBEDDING_DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS", "3072"))
# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX_DIR = os.path.join(BASE_DIR, "data", "faiss_index")

# ── Globals (loaded once at startup) ─────────────────────────────────────────
_faiss_index      = None
_train_df         = None
_bm25             = None
_tokenized_corpus = None
_gemini_model     = None


def load_all(index_dir: str = INDEX_DIR) -> None:
    """
    Load all models and indexes into memory. Call once at startup.
    Subsequent calls are no-ops.

    Args:
        index_dir: Path to the faiss_index folder from Milestone 2.
                   NOTE: if you rebuilt the index with 768-dim Gemini embeddings,
                   point this to the new index folder.
    """
    global _faiss_index, _train_df, _bm25, _tokenized_corpus, _gemini_model

    if _faiss_index is not None:
        return  # already loaded

    if not GEMINI_API_KEY:
        raise ValueError(
            "GEMINI_API_KEY not found. "
            "Make sure your .env file exists and contains GEMINI_API_KEY=..."
        )

    print("[RAG] Configuring Gemini API...")
    genai.configure(api_key=GEMINI_API_KEY)
    _gemini_model = genai.GenerativeModel(GEMINI_MODEL_NAME)
    print(f"[RAG] Gemini model ready: {GEMINI_MODEL_NAME}")

    print("[RAG] Loading FAISS index...")
    _faiss_index = faiss.read_index(os.path.join(index_dir, "train_index.faiss"))
    print(f"[RAG] Index loaded: {_faiss_index.ntotal:,} vectors, dim={_faiss_index.d}")

    # Warn if dimension mismatch — means index needs rebuilding with Gemini embeddings
    if _faiss_index.d != EMBEDDING_DIMENSIONS:
        print(
            f"[RAG] WARNING: Index dimension ({_faiss_index.d}) does not match "
            f"Gemini embedding dimension ({EMBEDDING_DIMENSIONS}). "
            f"Run rebuild_index() to fix this before searching."
        )

    print("[RAG] Loading train lookup table...")
    _train_df = pd.read_csv(os.path.join(index_dir, "train_lookup.csv"))

    print("[RAG] Loading BM25 corpus...")
    with open(os.path.join(index_dir, "bm25_corpus.pkl"), "rb") as f:
        _tokenized_corpus = pickle.load(f)
    _bm25 = BM25Okapi(_tokenized_corpus)

    print("[RAG] All components loaded. Ready.\n")


def rebuild_index(index_dir: str = INDEX_DIR, batch_size: int = 50) -> None:
    """
    Rebuild the FAISS index using Gemini text-embedding-004 (768-dim).

    Call this once after switching from sentence-transformers (384-dim)
    to Gemini embeddings (768-dim). Saves the new index to index_dir.

    Args:
        index_dir  : Where to save the rebuilt index
        batch_size : Rows per embedding API call (keep low to avoid rate limits)
    """
    global _faiss_index

    if _train_df is None:
        raise RuntimeError("Call load_all() first to load the lookup table.")

    print(f"[RAG] Rebuilding FAISS index with {EMBEDDING_MODEL_NAME} ({EMBEDDING_DIMENSIONS}-dim)...")
    print(f"[RAG] Embedding {len(_train_df):,} rows in batches of {batch_size}...")
    print("[RAG] This takes ~5-10 minutes on the free tier (rate limit: 100 RPM).\n")

    texts = (
        _train_df["instruction_clean"].fillna("") + " [SEP] " +
        _train_df["response_clean"].fillna("")
    ).tolist()

    all_embeddings = []
    import time
    from tqdm.auto import tqdm

    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch = texts[i:i + batch_size]
        result = genai.embed_content(
            model="models/gemini-embedding-001",
            content=batch,
            task_type="retrieval_document",
        )
        all_embeddings.extend(result["embedding"])
        time.sleep(0.6)   # stay within 100 RPM free tier limit

    embeddings = np.array(all_embeddings, dtype="float32")

    # L2-normalize so inner product = cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.maximum(norms, 1e-10)

    # Build and save new index
    index = faiss.IndexFlatIP(EMBEDDING_DIMENSIONS)
    index.add(embeddings)
    faiss.write_index(index, os.path.join(index_dir, "train_index.faiss"))
    np.save(os.path.join(index_dir, "train_embeddings.npy"), embeddings)

    _faiss_index = index
    print(f"\n[RAG] Index rebuilt: {index.ntotal:,} vectors saved to {index_dir}")


# ── Internal helpers ──────────────────────────────────────────────────────────

def _simple_tokenize(text: str) -> list:
    """Tokenize text for BM25 — lowercase words only."""
    return re.findall(r'\b\w+\b', text.lower())


def _embed_query(query: str) -> np.ndarray:
    """
    Embed a single query string using Gemini text-embedding-004.

    Uses task_type='retrieval_query' (different from 'retrieval_document'
    used when indexing) — Google recommends this split for best retrieval quality.

    Args:
        query: User's natural language question

    Returns:
        np.ndarray of shape (1, 768), float32, L2-normalized
    """
    result = genai.embed_content(
        model="models/gemini-embedding-001",
        content=query,
        task_type="retrieval_query",
    )
    vec = np.array(result["embedding"], dtype="float32").reshape(1, -1)
    vec = vec / np.maximum(np.linalg.norm(vec), 1e-10)
    return vec


def _retrieve_vector(query: str, top_k: int = 3) -> list:
    """Pure vector search via FAISS."""
    query_vec = _embed_query(query)
    scores, indices = _faiss_index.search(query_vec, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        row = _train_df.iloc[idx]
        results.append({
            "score"      : float(score),
            "category"   : row["category"],
            "intent"     : row["intent"],
            "instruction": row["instruction_clean"],
            "response"   : row["response_clean"],
        })
    return results


def _retrieve_hybrid(query: str, top_k: int = 3, alpha: float = 0.6) -> list:
    """
    Hybrid retrieval: fused Gemini vector + BM25 keyword scores.

    Args:
        query : User question
        top_k : Number of results
        alpha : Vector weight. 0.6 = slightly favour semantic over keyword.
    """
    n = len(_tokenized_corpus)

    # Vector scores
    query_vec = _embed_query(query)
    vec_scores, vec_indices = _faiss_index.search(query_vec, n)
    vec_score_map = dict(zip(vec_indices[0], vec_scores[0]))

    # BM25 scores normalized to [0, 1]
    bm25_scores = _bm25.get_scores(_simple_tokenize(query))
    bm25_max    = bm25_scores.max() if bm25_scores.max() > 0 else 1.0
    bm25_norm   = bm25_scores / bm25_max

    fused = np.array([
        alpha * vec_score_map.get(i, 0.0) + (1 - alpha) * bm25_norm[i]
        for i in range(n)
    ])
    top_indices = np.argsort(fused)[::-1][:top_k]

    results = []
    for idx in top_indices:
        row = _train_df.iloc[idx]
        results.append({
            "score"      : float(fused[idx]),
            "category"   : row["category"],
            "intent"     : row["intent"],
            "instruction": row["instruction_clean"],
            "response"   : row["response_clean"],
        })
    return results


def _build_prompt(query: str, context_docs: list) -> str:
    """
    Build the prompt sent to Gemini.

    Format:
        System: you are a helpful customer support agent...
        Context 1 [intent]: Q: ... A: ...
        Context 2 [intent]: Q: ... A: ...
        Customer question: ...
        Answer:
    """
    context_lines = []
    for i, doc in enumerate(context_docs, 1):
        context_lines.append(
            f"Context {i} [{doc['intent']}]:\n"
            f"  Q: {doc['instruction']}\n"
            f"  A: {doc['response']}"
        )
    context_block = "\n\n".join(context_lines)

    return (
        "You are a professional customer support agent. "
        "Use the context below to answer the customer's question accurately, "
        "concisely, and in a polite and helpful tone. "
        "If the context does not contain enough information to answer, "
        "say so politely and suggest the customer contact support directly.\n\n"
        f"{context_block}\n\n"
        f"Customer question: {query}\n\n"
        "Answer:"
    )


def _generate(prompt: str) -> str:
    """
    Generate an answer using Gemini 2.0 Flash.

    Args:
        prompt: Full prompt string from _build_prompt()

    Returns:
        Generated answer string
    """
    response = _gemini_model.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(
            temperature=0.2,       # low temp = more factual, less creative
            max_output_tokens=300,
        ),
    )
    return response.text.strip()


# ── Public API (called by notebook and FastAPI) ───────────────────────────────

def ask(
    query: str,
    top_k: int = 3,
    use_hybrid: bool = True,
) -> dict:
    """
    Full RAG pipeline: retrieve context, generate answer with Gemini.

    Args:
        query      : Customer's question (raw natural language)
        top_k      : Number of context docs to retrieve
        use_hybrid : True = vector+BM25 fusion, False = pure vector

    Returns:
        dict with keys: query, answer, sources, retrieval

    Example:
        >>> from src.rag_chain import load_all, ask
        >>> load_all()
        >>> result = ask("I want to cancel my order")
        >>> print(result["answer"])
    """
    if _faiss_index is None:
        raise RuntimeError("Models not loaded. Call load_all() first.")

    context_docs = _retrieve_hybrid(query, top_k=top_k) if use_hybrid \
                   else _retrieve_vector(query, top_k=top_k)

    prompt = _build_prompt(query, context_docs)
    answer = _generate(prompt)

    return {
        "query"    : query,
        "answer"   : answer,
        "sources"  : context_docs,
        "retrieval": "hybrid" if use_hybrid else "vector",
    }


def search(query: str, top_k: int = 5, use_hybrid: bool = True) -> list:
    """
    Retrieval only — no generation. Returns raw context docs.
    Used by the GET /search API endpoint.

    Args:
        query      : Natural language search query
        top_k      : Number of results
        use_hybrid : True = hybrid, False = vector only
    """
    if _faiss_index is None:
        raise RuntimeError("Models not loaded. Call load_all() first.")

    return _retrieve_hybrid(query, top_k=top_k) if use_hybrid \
           else _retrieve_vector(query, top_k=top_k)
