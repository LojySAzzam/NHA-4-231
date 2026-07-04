"""
rag_chain.py
------------
Milestone 3 — Core RAG logic (Gemini 2.0 Flash + gemini-embedding-001)

Pipeline:
    user query
        -> embed query (Google gemini-embedding-001, 3072-dim)
        -> retrieve top-k context docs (FAISS)
        -> build prompt with context
        -> generate answer (Gemini 2.0 Flash via Google AI Studio)
        -> return answer + sources
"""

import os
import re
import time
import pickle
import numpy as np
import pandas as pd
import faiss
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
import google.generativeai as genai
from tqdm.auto import tqdm

# ── Load environment variables ────────────────────────────────────────────────
load_dotenv()

GEMINI_API_KEY       = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL_NAME    = os.getenv("GEMINI_MODEL_NAME",    "gemini-2.0-flash")
EMBEDDING_MODEL_NAME = "gemini-embedding-001"
EMBEDDING_DIMENSIONS = 3072

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX_DIR = os.path.join(BASE_DIR, "data", "faiss_index")

# ── Globals ───────────────────────────────────────────────────────────────────
_faiss_index      = None
_train_df         = None
_bm25             = None
_tokenized_corpus = None
_gemini_model     = None


def load_all(index_dir: str = INDEX_DIR) -> None:
    """
    Load all models and indexes into memory. Call once at startup.
    Subsequent calls are no-ops.
    """
    global _faiss_index, _train_df, _bm25, _tokenized_corpus, _gemini_model

    if _faiss_index is not None:
        return

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

    if _faiss_index.d != EMBEDDING_DIMENSIONS:
        print(
            f"[RAG] WARNING: Index dimension ({_faiss_index.d}) does not match "
            f"Gemini embedding dimension ({EMBEDDING_DIMENSIONS}). "
            f"Run rebuild_index() before searching."
        )

    print("[RAG] Loading train lookup table...")
    _train_df = pd.read_csv(os.path.join(index_dir, "train_lookup.csv"))

    print("[RAG] Loading BM25 corpus...")
    with open(os.path.join(index_dir, "bm25_corpus.pkl"), "rb") as f:
        _tokenized_corpus = pickle.load(f)
    _bm25 = BM25Okapi(_tokenized_corpus)

    print("[RAG] All components loaded. Ready.\n")


def rebuild_index(index_dir: str = INDEX_DIR, batch_size: int = 5) -> None:
    """
    Rebuild the FAISS index using gemini-embedding-001 (3072-dim).
    Run once after switching from sentence-transformers (384-dim).
    Saves the new index to index_dir, overwriting the old one.
    Supports resuming from a checkpoint if interrupted by rate limits.
    """
    global _faiss_index

    if _train_df is None:
        raise RuntimeError("Call load_all() first to load the lookup table.")

    print(f"[RAG] Rebuilding FAISS index with {EMBEDDING_MODEL_NAME} ({EMBEDDING_DIMENSIONS}-dim)...")
    print(f"[RAG] Embedding {len(_train_df):,} rows in batches of {batch_size}...")
    print("[RAG] Rate limit: ~30 RPM (conservative). Checkpoints every 500 rows.\n")

    texts = (
        _train_df["instruction_clean"].fillna("") + " [SEP] " +
        _train_df["response_clean"].fillna("")
    ).tolist()

    checkpoint_file = os.path.join(index_dir, "embeddings_checkpoint.npy")

    # Resume from checkpoint if it exists
    if os.path.exists(checkpoint_file):
        all_embeddings = list(np.load(checkpoint_file))
        start_i = len(all_embeddings)
        # Round down to nearest batch boundary
        start_i = (start_i // batch_size) * batch_size
        all_embeddings = all_embeddings[:start_i]
        print(f"[RAG] Resuming from row {start_i} ({len(all_embeddings)} embeddings already done)\n")
    else:
        all_embeddings = []
        start_i = 0

    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch = texts[i:i + batch_size]

        for attempt in range(6):
            try:
                result = genai.embed_content(
                    model=f"models/{EMBEDDING_MODEL_NAME}",
                    content=batch,
                    task_type="retrieval_document",
                )
                all_embeddings.extend(result["embedding"])
                time.sleep(10)  # conservative: ~30 RPM, well under 100 RPM limit
                break
            except Exception as e:
                if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e):
                    wait = 60 + (attempt * 15)
                    print(f"\nRate limit hit — waiting {wait}s (attempt {attempt+1}/6)...")
                    time.sleep(wait)
                else:
                    raise
        else:
            # All 6 attempts failed — save checkpoint and stop gracefully
            np.save(checkpoint_file, np.array(all_embeddings, dtype="float32"))
            print(f"\n[RAG] Quota exhausted. Saved checkpoint at {len(all_embeddings)} embeddings.")
            print("[RAG] Run rebuild_index() again tomorrow to resume automatically.")
            return
        
        # Save checkpoint every 500 embeddings
        if len(all_embeddings) % 500 == 0:
            np.save(checkpoint_file, np.array(all_embeddings, dtype="float32"))
            print(f"\n[RAG] Checkpoint saved at {len(all_embeddings)} embeddings.")

    
    if not all_embeddings:
        raise RuntimeError("No embeddings were generated. Check your API key and quota.")

    embeddings = np.array(all_embeddings, dtype="float32")

    # L2-normalize so inner product == cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.maximum(norms, 1e-10)

    # Build and save new FAISS index
    index = faiss.IndexFlatIP(EMBEDDING_DIMENSIONS)
    index.add(embeddings)
    faiss.write_index(index, os.path.join(index_dir, "train_index.faiss"))
    np.save(os.path.join(index_dir, "train_embeddings.npy"), embeddings)

    # Clean up checkpoint file now that we're done
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        print("[RAG] Checkpoint file removed.")

    _faiss_index = index
    print(f"\n[RAG] Done: {index.ntotal:,} vectors saved to {index_dir}")


# ── Internal helpers ──────────────────────────────────────────────────────────

def _simple_tokenize(text: str) -> list:
    return re.findall(r'\b\w+\b', text.lower())


def _embed_query(query: str) -> np.ndarray:
    """Embed a single query using gemini-embedding-001 with retrieval_query task."""
    result = genai.embed_content(
        model=f"models/{EMBEDDING_MODEL_NAME}",
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
    """Hybrid retrieval: fused Gemini vector + BM25 keyword scores."""
    n = len(_tokenized_corpus)

    query_vec = _embed_query(query)
    vec_scores, vec_indices = _faiss_index.search(query_vec, n)
    vec_score_map = dict(zip(vec_indices[0], vec_scores[0]))

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
    """Build the prompt sent to Gemini."""
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
        "If the context does not contain enough information, "
        "say so politely and suggest the customer contact support directly.\n\n"
        f"{context_block}\n\n"
        f"Customer question: {query}\n\n"
        "Answer:"
    )


def _generate(prompt: str) -> str:
    """Generate an answer using Gemini 2.0 Flash."""
    response = _gemini_model.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(
            temperature=0.2,
            max_output_tokens=300,
        ),
    )
    return response.text.strip()


# ── Public API ────────────────────────────────────────────────────────────────

def ask(query: str, top_k: int = 3, use_hybrid: bool = True) -> dict:
    """
    Full RAG pipeline: retrieve context, generate answer with Gemini.

    Args:
        query      : Customer question
        top_k      : Number of context docs to retrieve
        use_hybrid : True = vector+BM25, False = pure vector

    Returns:
        dict with keys: query, answer, sources, retrieval
    """
    if _faiss_index is None:
        raise RuntimeError("Models not loaded. Call load_all() first.")

    context_docs = (
        _retrieve_hybrid(query, top_k=top_k)
        if use_hybrid
        else _retrieve_vector(query, top_k=top_k)
    )

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
    """
    if _faiss_index is None:
        raise RuntimeError("Models not loaded. Call load_all() first.")

    return (
        _retrieve_hybrid(query, top_k=top_k)
        if use_hybrid
        else _retrieve_vector(query, top_k=top_k)
    )
