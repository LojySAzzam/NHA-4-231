"""
rag_chain.py
------------
Milestone 3 — Core RAG logic (refactored)
Embeddings : Google Gemini (gemini-embedding-001, 3072-dim)
Generation : Groq (llama-3.3-70b-versatile)

Pipeline:
    user query
        -> embed query (Gemini embedding API)
        -> retrieve top-k context docs (FAISS + BM25, fused via Reciprocal
           Rank Fusion instead of raw score blending)
        -> build prompt with context
        -> generate answer (Groq llama-3.3-70b-versatile)
        -> return answer + sources

Refactor notes (see project defense guide for full rationale):
    - _embed_query() de-duplicated to a single definition.
    - Hybrid fusion now uses RRF (rank-based) instead of alpha-weighted
      raw score blending, eliminating the FAISS-vs-BM25 scale mismatch.
    - rebuild_index() is now backed by an async, semaphore-bounded
      concurrent embedding pipeline (rebuild_index_async) instead of a
      fully sequential requests.post + time.sleep(10) loop.
    - Configuration is validated declaratively via pydantic_settings at
      import time (fail-fast, aggregated errors) instead of scattered
      `if not KEY: raise ValueError` checks inside load_all().
    - A public is_ready() function replaces direct external access to
      this module's private globals (e.g. api.py no longer reaches into
      rag._faiss_index).
"""

import os
import re
import asyncio
import pickle
import numpy as np
import pandas as pd
import faiss
import httpx
import requests
import time
from rank_bm25 import BM25Okapi
from tqdm.auto import tqdm
from groq import Groq
import google.generativeai as genai
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# ── Declarative configuration (fail-fast, aggregated validation) ─────────────

class Settings(BaseSettings):
    """
    Typed, validated configuration loaded from the environment / .env file.
    Instantiating this at module import time means a missing or invalid
    required variable raises one clear, aggregated pydantic ValidationError
    immediately on startup, rather than failing deep inside load_all() the
    first time a particular key happens to be checked.
    """
    model_config = SettingsConfigDict(env_file=".env", populate_by_name=True, extra="ignore")

    gemini_api_key   : str            = Field(..., alias="GEMINI_API_KEY")
    gemini_api_key_2 : str | None     = Field(None, alias="GEMINI_API_KEY_2")
    groq_api_key     : str            = Field(..., alias="GROQ_API_KEY")
    groq_model_name  : str            = Field("llama-3.3-70b-versatile", alias="GROQ_MODEL_NAME")


settings = Settings()  # raises immediately at import time if required keys are missing

EMBEDDING_MODEL_NAME = "gemini-embedding-001"
EMBEDDING_DIMENSIONS = 3072
RRF_K                = 60  # standard Reciprocal Rank Fusion constant (Cormack et al.)

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

    print("[RAG] Configuring Gemini API for embeddings...")
    genai.configure(api_key=settings.gemini_api_key)
    print(f"[RAG] Gemini embedding model: {EMBEDDING_MODEL_NAME} ({EMBEDDING_DIMENSIONS}-dim)")

    print("[RAG] Initialising Groq client for generation...")
    _groq_client = Groq(api_key=settings.groq_api_key)
    print(f"[RAG] Groq ready: {settings.groq_model_name}")

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


def is_ready() -> bool:
    """
    Public readiness/liveness check, safe to call on every /health request
    (no network calls — fast, synchronous, side-effect free).

    Verifies all four runtime components:
        1. FAISS index is loaded
        2. BM25 tokenizer/index is loaded
        3. Groq client is initialised
        4. Required upstream API keys are present (validated once at
           import time by Settings; re-checked here defensively so a
           caller never needs to reach into this module's private globals)
    """
    return (
        _faiss_index is not None
        and _bm25 is not None
        and _groq_client is not None
        and bool(settings.gemini_api_key)
        and bool(settings.groq_api_key)
    )


# ── Async indexing pipeline ────────────────────────────────────────────────────

class _RateLimited(Exception):
    """Raised internally when Gemini returns HTTP 429 for a batch."""


class _UpstreamUnavailable(Exception):
    """Raised internally when Gemini returns a transient 5xx for a batch."""


async def _embed_batch_async(client: httpx.AsyncClient, batch: list[str], api_key: str, key_idx: int) -> list:
    """Single async batchEmbedContents call against the Gemini REST API."""
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{EMBEDDING_MODEL_NAME}:batchEmbedContents?key={api_key}"
    )
    payload = {
        "requests": [
            {
                "model": f"models/{EMBEDDING_MODEL_NAME}",
                "content": {"parts": [{"text": text}]},
                "taskType": "RETRIEVAL_DOCUMENT",
            }
            for text in batch
        ]
    }
    resp = await client.post(url, json=payload, timeout=30.0)

    # ── NEW: Explicitly print the active key tracker and status code ──
    print(f"\n[RAG] Key Index {key_idx} -> Status Code: {resp.status_code}")

    if resp.status_code == 200:
        data = resp.json()
        return [item["values"] for item in data["embeddings"]]
    if resp.status_code == 429:
        raise _RateLimited(resp.status_code)
    if resp.status_code in (500, 502, 503, 504):
        raise _UpstreamUnavailable(resp.status_code)
    raise RuntimeError(f"Google API Error {resp.status_code}: {resp.text}")


async def _embed_batch_with_retry(
    client: httpx.AsyncClient,
    batch: list[str],
    api_keys: list[str],
    sem: asyncio.Semaphore,
    key_cursor: list[int],
) -> list | None:
    """
    Embed one batch, rotating across every available API key before
    backing off — mirrors the original synchronous design:
        - 429 on one key -> immediately try the next key (no wait)
        - 429 on ALL keys -> burn one "attempt" and back off
          (60 + attempt * 15 seconds), up to 6 attempts
        - transient 5xx -> short fixed retry, doesn't count as an attempt

    Returns None if every key is exhausted after 6 backoff attempts, so the
    caller can checkpoint and stop gracefully for the day (same contract as
    the original rebuild_index loop).
    """
    async with sem:
        for attempt in range(6):
            # ── CHANGED: Read the current key directly without looping through all of them ──
            key_idx = key_cursor[0] % len(api_keys)
            api_key = api_keys[key_idx]
            try:
                # Pass key_idx down to the printer function
                return await _embed_batch_async(client, batch, api_key, key_idx)
            except _RateLimited:
                # Move the cursor to the next key so the next batch uses it
                key_cursor[0] += 1
                # If you only have 1 key left or both are failing, apply backoff immediately
                wait = 60 + attempt * 15
                print(f"\n[RAG] Key Index {key_idx} rate limited (429). "
                      f"Switched pointer. Backing off for {wait}s (attempt {attempt + 1}/6)...")
                await asyncio.sleep(wait)
                continue  # Retry this batch on the next attempt with the new key

            except _UpstreamUnavailable:
                await asyncio.sleep(15)
                continue

        return None


async def rebuild_index_async(
    texts: list[str],
    api_keys: list[str],
    batch_size: int = 5,
    max_concurrent: int = 4,
) -> tuple[np.ndarray | None, int]:
    """
    Concurrently embed `texts` via Gemini batchEmbedContents.

    Replaces the fully sequential `requests.post` + `time.sleep(10)` loop
    with a semaphore-bounded (`max_concurrent`, default 4) concurrent
    pipeline, so throughput scales with allowed concurrency instead of
    being capped at one in-flight request at a time — while still
    respecting Gemini's rate limits via the semaphore and the same
    exponential backoff (60 + attempt * 15s) on sustained 429s.

    Returns:
        (embeddings, n_completed_batches)
        `embeddings` covers only the leading contiguous run of batches that
        succeeded before any exhaustion event, preserving the original
        "resume from row N" checkpoint semantics.
    """
    batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
    if not batches:
        return None, 0

    key_cursor = [0]
    sem = asyncio.Semaphore(max_concurrent)
    pbar = tqdm(total=len(batches), desc="Embedding")

    async with httpx.AsyncClient() as client:
        async def _tracked(batch: list[str]):
            result = await _embed_batch_with_retry(client, batch, api_keys, sem, key_cursor)
            pbar.update(1)
            return result

        # gather() preserves input order in its results regardless of which
        # batch actually finishes first, so `ordered[i]` always corresponds
        # to `batches[i]` — required for the contiguous-prefix checkpoint
        # logic below to be correct under concurrent execution.
        ordered = await asyncio.gather(*(_tracked(b) for b in batches))

    pbar.close()

    n_completed = 0
    for batch_result in ordered:
        if batch_result is None:
            break  # stop at the first exhausted batch to keep a contiguous prefix
        n_completed += 1

    if n_completed == 0:
        return None, 0

    flat = [vec for batch in ordered[:n_completed] for vec in batch]
    return np.array(flat, dtype="float32"), n_completed


def rebuild_index(index_dir: str = INDEX_DIR, batch_size: int = 5, max_concurrent: int = 4) -> None:
    """
    Rebuild the FAISS index using Gemini gemini-embedding-001 (3072-dim),
    driven by the concurrent, semaphore-bounded rebuild_index_async().
    Synchronous implementation using robust key rotation, exponential backoff,
    and structured on-disk checkpointing every 500 processed rows.

    Supports:
        - Checkpoint resume (safe to re-run if interrupted)
        - Automatic key rotation between GEMINI_API_KEY and GEMINI_API_KEY_2
        - Rate limit handling with exponential backoff (60 + attempt*15s)
    """
    global _faiss_index

    if _train_df is None:
        raise RuntimeError("Call load_all() first.")
    
    # ── Load both keys out of Pydantic configuration settings ──

    api_keys = [k for k in [settings.gemini_api_key, settings.gemini_api_key_2] if k]
    if not api_keys:
        raise ValueError("No Gemini API keys configured.")
    

    print(f"[RAG] Using {len(api_keys)} API key(s) for embedding,"
          f"max_concurrent={max_concurrent}.")
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

    last_status_msg = ""

    remaining_texts = texts[start_i:]
    if not remaining_texts:
        print("[RAG] Nothing left to embed.")
        return

    # ── THE ASYNC LINE: High-throughput concurrent execution ──
    # This runs the high-speed batching engine inside your clean synchronous function block
    new_embeddings, n_completed_batches = asyncio.run(
        rebuild_index_async(
            remaining_texts, 
            api_keys, 
            batch_size=batch_size, 
            max_concurrent=max_concurrent
        )
    )

    if new_embeddings is None:
        print("[RAG] No new embeddings could be generated this run "
              "(quota exhausted immediately). Run rebuild_index() again later.")
        return

   # Combine existing checkpoint embeddings with the newly computed async embeddings
    all_embeddings.extend(new_embeddings.tolist())
    n_completed_rows = start_i + (n_completed_batches * batch_size)

    # ── Exact 500-Row Hard Checkpoint Verification ──
    # If the async loop finishes a chunk or gets paused, we log the precise snapshot to disk

    if len(all_embeddings) % 500 == 0 and len(all_embeddings) > 0:
           np.save(checkpoint_file, np.array(all_embeddings, dtype="float32"))
           print(f"[RAG] Checkpoint saved at {len(all_embeddings)} embeddings.")
   
    if n_completed_rows < len(texts):
        np.save(checkpoint_file, np.array(all_embeddings, dtype="float32"))

        print(f"\n[RAG] Stream paused or quota reached after {n_completed_rows:,}/{len(texts):,} rows.")
        print("[RAG] Checkpoint saved successfully. Run rebuild_index() again to resume.")
        return
    
    # Full run complete — normalize, build index, persist, clear checkpoint.
    embeddings = np.array(all_embeddings, dtype="float32")
    norms = np.linalg.norm(all_embeddings, axis=1, keepdims=True)
    all_embeddings = all_embeddings / np.maximum(norms, 1e-10)

    index = faiss.IndexFlatIP(EMBEDDING_DIMENSIONS)
    index.add(all_embeddings)
    faiss.write_index(index, os.path.join(index_dir, "train_index.faiss"))
    np.save(os.path.join(index_dir, "train_embeddings.npy"), all_embeddings)

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

    Returns None if the embedding quota is exhausted, instead of crashing —
    this is a deliberate, non-retrying fast-fail: a live HTTP request
    cannot tolerate the multi-attempt exponential backoff used during bulk
    indexing, so the caller (_retrieve_hybrid) degrades to a BM25-only RRF
    ranking immediately rather than stalling the request.
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


def _retrieve_hybrid(query: str, top_k: int = 3, k_rrf: int = RRF_K) -> list:
    """
    Hybrid retrieval via Reciprocal Rank Fusion (RRF).

    Replaces the previous `alpha * FAISS + (1 - alpha) * BM25_norm` blend,
    which suffered from scale distortion (FAISS cosine similarity lives in
    [-1, 1]; BM25 raw scores are unbounded and were max-normalized *per
    query*, so the same document's contribution shifted depending on what
    else was in the corpus for that query). RRF instead fuses purely on
    rank position, so no cross-normalization is ever needed:

        score(d) = sum over each ranker r of 1 / (k_rrf + rank_r(d))

    Falls back to a BM25-only RRF ranking if the Gemini embedding quota is
    exhausted. This preserves the original fallback contract exactly:
    the online query path never retries or backs off on 429 — it fails
    fast and re-ranks with whatever rankers are actually available, so a
    live request is never stalled waiting on Gemini.
    """
    query_vec = _embed_query(query)

    dense_ranked_ids = None
    if query_vec is not None:
        _, indices = _faiss_index.search(query_vec, len(_tokenized_corpus))
        dense_ranked_ids = indices[0]
    else:
        print("[RAG] Embedding quota exhausted — using BM25-only RRF fallback for retrieval.")

    bm25_scores = _bm25.get_scores(_simple_tokenize(query))
    sparse_ranked_ids = np.argsort(bm25_scores)[::-1]

    rrf_scores: dict = {}
    if dense_ranked_ids is not None:
        for rank, idx in enumerate(dense_ranked_ids):
            idx = int(idx)
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (k_rrf + rank + 1)
    for rank, idx in enumerate(sparse_ranked_ids):
        idx = int(idx)
        rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (k_rrf + rank + 1)

    top_indices = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:top_k]

    results = []
    for idx in top_indices:
        row = _train_df.iloc[idx]
        results.append({
            "score"      : float(rrf_scores[idx]),
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
    """Generate an answer using Groq."""
    response = _groq_client.chat.completions.create(
        model=settings.groq_model_name,
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
        use_hybrid : True = RRF-fused vector+BM25, False = pure vector

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

# ── Expose Pydantic settings attributes to module root for notebook compatibility ──

# This instantiates your class if it hasn't been instantiated, or uses the global instance
GROQ_MODEL_NAME = settings.groq_model_name