"""
rag_chain.py
------------
Milestone 3 — Core RAG logic
Embeddings : Google Gemini (gemini-embedding-001, 3072-dim)
Generation : Groq (llama-3.3-70b-versatile)

Pipeline:
    user query
        -> embed query (Gemini embedding API)
        -> retrieve top-k context docs (FAISS hybrid search)
        -> build prompt with context
        -> generate answer (Groq llama-3.3-70b-versatile)
        -> return answer + sources
"""

import os
import re
import time
import pickle
import numpy as np
import pandas as pd
import faiss
import requests
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from tqdm.auto import tqdm
from groq import Groq
import google.generativeai as genai

# ── Load environment variables ────────────────────────────────────────────────
load_dotenv()

GEMINI_API_KEY       = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY         = os.getenv("GROQ_API_KEY")
GROQ_MODEL_NAME      = os.getenv("GROQ_MODEL_NAME", "llama-3.3-70b-versatile")
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
_groq_client      = None


def load_all(index_dir: str = INDEX_DIR) -> None:
    """
    Load all models and indexes into memory. Call once at startup.
    Subsequent calls are no-ops.
    """
    global _faiss_index, _train_df, _bm25, _tokenized_corpus, _groq_client

    if _faiss_index is not None:
        return

    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not found in .env file.")
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not found in .env file.")

    print("[RAG] Configuring Gemini API for embeddings...")
    genai.configure(api_key=GEMINI_API_KEY)
    print(f"[RAG] Gemini embedding model: {EMBEDDING_MODEL_NAME} ({EMBEDDING_DIMENSIONS}-dim)")

    print("[RAG] Initialising Groq client for generation...")
    _groq_client = Groq(api_key=GROQ_API_KEY)
    print(f"[RAG] Groq ready: {GROQ_MODEL_NAME}")

    print("[RAG] Loading FAISS index...")
    _faiss_index = faiss.read_index(os.path.join(index_dir, "train_index.faiss"))
    print(f"[RAG] Index loaded: {_faiss_index.ntotal:,} vectors, dim={_faiss_index.d}")

    if _faiss_index.d != EMBEDDING_DIMENSIONS:
        print(
            f"[RAG] WARNING: Index dim ({_faiss_index.d}) != "
            f"embedding dim ({EMBEDDING_DIMENSIONS}). "
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
    Rebuild the FAISS index using Gemini gemini-embedding-001 (3072-dim).
    Supports:
        - Checkpoint resume (safe to re-run if interrupted)
        - Automatic key rotation between GEMINI_API_KEY and GEMINI_API_KEY_2
        - Rate limit handling with exponential backoff

    Daily capacity:
        1000 requests/key x 2 keys x 5 rows/batch = 10,000 rows/day
        Full 17K dataset completes in 2 days.
    """
    global _faiss_index

    if _train_df is None:
        raise RuntimeError("Call load_all() first.")

    # Load both keys for rotation
    api_keys = [k for k in [
        os.getenv("GEMINI_API_KEY"),
        os.getenv("GEMINI_API_KEY_2"),
    ] if k]

    if not api_keys:
        raise ValueError("No Gemini API keys found in .env file.")

    current_key_idx = 0
    genai.configure(api_key=api_keys[current_key_idx])
    print(f"[RAG] Using {len(api_keys)} API key(s) for embedding.")

    print(f"[RAG] Rebuilding FAISS index with {EMBEDDING_MODEL_NAME} ({EMBEDDING_DIMENSIONS}-dim)...")
    print(f"[RAG] {len(_train_df):,} rows, batch_size={batch_size}")
    print("[RAG] Daily capacity: 1000 req/key x 5 rows = 5000 rows/key/day")
    print("[RAG] Checkpoint enabled — safe to re-run if interrupted.\n")

    texts = (
        _train_df["instruction_clean"].fillna("") + " [SEP] " +
        _train_df["response_clean"].fillna("")
    ).tolist()

    checkpoint_file = os.path.join(index_dir, "embeddings_checkpoint.npy")

    # Resume from checkpoint if exists
    if os.path.exists(checkpoint_file):
        all_embeddings = list(np.load(checkpoint_file))
        start_i = (len(all_embeddings) // batch_size) * batch_size
        all_embeddings = all_embeddings[:start_i]
        print(f"[RAG] Resuming from row {start_i} ({len(all_embeddings)} embeddings done)\n")
    else:
        all_embeddings = []
        start_i = 0

    print(f"[DEBUG] Total texts to embed: {len(texts)}")
    print(f"[DEBUG] Starting loop at index i = {start_i}")
    
    # --- NEW: Variable to track the last printed status ---
    last_status_msg = ""
    
    for i in tqdm(range(start_i, len(texts), batch_size), desc="Embedding"):
        batch = texts[i:i + batch_size]

        batch_success = False

        # Loop 1: The Exponential Backoff (Only triggers when ALL keys fail)
        for attempt in range(6):
            if batch_success:
                break

            # Loop 2: The Key Rotation (Tries every key before giving up)
            for _ in range(len(api_keys)):
                key_to_use = api_keys[current_key_idx]
                
                # Direct REST API call to completely bypass the SDK's caching issues
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{EMBEDDING_MODEL_NAME}:batchEmbedContents?key={key_to_use}"
                
                # Format the payload for batch embedding
                payload = {
                    "requests": [
                        {
                            "model": f"models/{EMBEDDING_MODEL_NAME}",
                            "content": {"parts": [{"text": text}]},
                            "taskType": "RETRIEVAL_DOCUMENT"
                        } for text in batch
                    ]
                }

                response = requests.post(url, json=payload, headers={"Content-Type": "application/json"})

                current_status_msg = f"[RAG] Key Index {current_key_idx} -> Status Code: {response.status_code}"
                
                if current_status_msg != last_status_msg:
                    print(f"\n{current_status_msg}")
                    last_status_msg = current_status_msg

                if response.status_code == 200:
                    data = response.json()
                    
                    # Extract the vector arrays from the JSON response
                    batch_embeddings = [item["values"] for item in data["embeddings"]]
                    all_embeddings.extend(batch_embeddings) 
                    time.sleep(10)  # ~30 RPM, well under 100 RPM limit
                    batch_success = True
                    break  # Break out of the inner key-rotation loop

                elif response.status_code == 429:
                    # Rate limit hit! Swap to the next key immediately.
                    current_key_idx = (current_key_idx + 1) % len(api_keys)
                    print(f"\n[RAG] Key exhausted (429) — dynamically switched to key index {current_key_idx}")
                    continue  # Try the next key in the inner loop

                elif response.status_code in [500, 502, 503, 504]:
                    # Google's servers are temporarily down or overloaded
                    print(f"\n[RAG] Server Error ({response.status_code}). Google is hiccuping. Retrying in 15s...")
                    time.sleep(15)
                    continue # Let the loop automatically try again

                else:
                    # If it's a 400 Bad Request or 500 error, crash loudly
                    raise Exception(f"Google API Error {response.status_code}: {response.text}")

            if batch_success:
                break  # Break out of the attempt loop and move to the next batch of rows
            
            # If we reach here, we looped through ALL keys and they ALL returned 429s.
            # Now, and only now, do we burn an 'attempt' and apply the exponential backoff.
            wait = 60 + (attempt * 15)
            print(f"\n[RAG] All keys exhausted — waiting {wait}s (attempt {attempt+1}/6)...")
            time.sleep(wait)

        if not batch_success:
            # All 6 attempts failed — save checkpoint and stop gracefully
            np.save(checkpoint_file, np.array(all_embeddings, dtype="float32"))
            print(f"\n[RAG] Both keys exhausted for today.")
            print(f"[RAG] Checkpoint saved at {len(all_embeddings)} embeddings ({len(all_embeddings)} rows done).")
            print("[RAG] Run rebuild_index() tomorrow — will resume automatically from here.")
            return

        # Save checkpoint every 500 embeddings
        if len(all_embeddings) % 500 == 0 and len(all_embeddings) > 0:
            np.save(checkpoint_file, np.array(all_embeddings, dtype="float32"))
            print(f"\n[RAG] Checkpoint saved at {len(all_embeddings)} embeddings.")

    if not all_embeddings:
        raise RuntimeError("No embeddings generated. Check your API keys and quota.")

    embeddings = np.array(all_embeddings, dtype="float32")

    # L2-normalize so inner product == cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.maximum(norms, 1e-10)

    # Build and save new FAISS index
    index = faiss.IndexFlatIP(EMBEDDING_DIMENSIONS)
    index.add(embeddings)
    faiss.write_index(index, os.path.join(index_dir, "train_index.faiss"))
    np.save(os.path.join(index_dir, "train_embeddings.npy"), embeddings)

    # Clean up checkpoint now that we're done
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        print("[RAG] Checkpoint file removed.")

    _faiss_index = index
    print(f"\n[RAG] Done: {index.ntotal:,} vectors saved to {index_dir}")


# ── Internal helpers ──────────────────────────────────────────────────────────

def _simple_tokenize(text: str) -> list:
    return re.findall(r'\b\w+\b', text.lower())


def _embed_query(query: str):
    """
    Embed a single query using Gemini gemini-embedding-001.
    Returns None if quota is exhausted instead of crashing.
    """
    try:
        result = genai.embed_content(
            model=f"models/{EMBEDDING_MODEL_NAME}",
            content=query,
            task_type="retrieval_query",
        )
        vec = np.array(result["embedding"], dtype="float32").reshape(1, -1)
        vec = vec / np.maximum(np.linalg.norm(vec), 1e-10)
        return vec
    except Exception as e:
        if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e):
            return None  # quota exhausted — caller will fall back to BM25
        raise
def _embed_query(query: str):
    """
    Embed a single query using Gemini gemini-embedding-001.
    Returns None if quota is exhausted instead of crashing.
    """
    try:
        result = genai.embed_content(
            model=f"models/{EMBEDDING_MODEL_NAME}",
            content=query,
            task_type="retrieval_query",
        )
        vec = np.array(result["embedding"], dtype="float32").reshape(1, -1)
        vec = vec / np.maximum(np.linalg.norm(vec), 1e-10)
        return vec
    except Exception as e:
        if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e):
            return None  # quota exhausted — caller will fall back to BM25
        raise
def _embed_query(query: str):
    """
    Embed a single query using Gemini gemini-embedding-001.
    Returns None if quota is exhausted instead of crashing.
    """
    try:
        result = genai.embed_content(
            model=f"models/{EMBEDDING_MODEL_NAME}",
            content=query,
            task_type="retrieval_query",
        )
        vec = np.array(result["embedding"], dtype="float32").reshape(1, -1)
        vec = vec / np.maximum(np.linalg.norm(vec), 1e-10)
        return vec
    except Exception as e:
        if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e):
            return None  # quota exhausted — caller will fall back to BM25
        raise

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
    Falls back to pure BM25 if Gemini embedding quota is exhausted.
    """
    n = len(_tokenized_corpus)

    query_vec = _embed_query(query)
    if query_vec is not None:
        vec_scores, vec_indices = _faiss_index.search(query_vec, n)
        vec_score_map = dict(zip(vec_indices[0], vec_scores[0]))
        bm25_scores = _bm25.get_scores(_simple_tokenize(query))
        bm25_max    = bm25_scores.max() if bm25_scores.max() > 0 else 1.0
        bm25_norm   = bm25_scores / bm25_max
        fused = np.array([
            alpha * vec_score_map.get(i, 0.0) + (1 - alpha) * bm25_norm[i]
            for i in range(n)
        ])
    else:
        # Fallback: pure BM25 (no embedding quota used)
        print("[RAG] Embedding quota exhausted — using BM25 fallback for retrieval.")
        fused = _bm25.get_scores(_simple_tokenize(query))
        
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
    """Build the prompt sent to Groq."""
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
    """Generate an answer using Groq llama-3.3-70b-versatile."""
    response = _groq_client.chat.completions.create(
        model=GROQ_MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=300,
    )
    return response.choices[0].message.content.strip()


# ── Public API ────────────────────────────────────────────────────────────────

def ask(query: str, top_k: int = 3, use_hybrid: bool = True) -> dict:
    """
    Full RAG pipeline: retrieve context, generate answer.

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
