"""
rag_chain.py
------------
Milestone 3 — Core RAG logic

This module is shared by both:
  - notebooks/Milestone_3_Local.ipynb  (interactive testing)
  - src/api.py                          (FastAPI production server)

Pipeline per query:
  1. Embed the user question with sentence-transformers
  2. Retrieve top-k nearest documents from the FAISS index
  3. Build a prompt: [system instructions] + [retrieved context] + [question]
  4. Generate an answer with flan-t5-base running locally on CPU
  5. Return the answer + the retrieved source documents
"""

import os
import re
import pickle
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from transformers import T5ForConditionalGeneration, T5Tokenizer
from rank_bm25 import BM25Okapi


# ── Paths ─────────────────────────────────────────────────────────────────────
# These resolve correctly whether called from notebooks/ or src/
_HERE       = os.path.dirname(os.path.abspath(__file__))
_ROOT       = os.path.dirname(_HERE)
INDEX_DIR   = os.path.join(_ROOT, "data", "faiss_index")

# ── Model names ───────────────────────────────────────────────────────────────
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"   # same model used in Milestone 2
LLM_MODEL_NAME   = "google/flan-t5-base" # 250MB, instruction-following, CPU-friendly

# ── Global model handles (loaded once, reused across requests) ────────────────
_embed_model  = None
_llm_model    = None
_llm_tokenizer = None
_faiss_index  = None
_lookup_df    = None
_bm25         = None
_train_texts  = None


def _load_models():
    """
    Load all models and indexes into module-level globals.
    Called automatically on first use — subsequent calls are no-ops.
    """
    global _embed_model, _llm_model, _llm_tokenizer
    global _faiss_index, _lookup_df, _bm25, _train_texts

    if _embed_model is not None:
        return  # already loaded

    print("[RAG] Loading embedding model...")
    _embed_model = SentenceTransformer(EMBED_MODEL_NAME)

    print("[RAG] Loading FAISS index...")
    _faiss_index = faiss.read_index(os.path.join(INDEX_DIR, "train_index.faiss"))
    _lookup_df   = pd.read_csv(os.path.join(INDEX_DIR, "train_lookup.csv"))

    print("[RAG] Loading BM25 index...")
    with open(os.path.join(INDEX_DIR, "bm25_corpus.pkl"), "rb") as f:
        tokenized_corpus = pickle.load(f)
    _bm25 = BM25Okapi(tokenized_corpus)

    # Rebuild raw texts for BM25 scoring (used in hybrid search)
    _train_texts = (
        _lookup_df["instruction_clean"].fillna("") +
        " [SEP] " +
        _lookup_df["response_clean"].fillna("")
    ).tolist()

    print("[RAG] Loading LLM (flan-t5-base)... first run downloads ~250MB")
    _llm_tokenizer = T5Tokenizer.from_pretrained(LLM_MODEL_NAME)
    _llm_model     = T5ForConditionalGeneration.from_pretrained(LLM_MODEL_NAME)
    _llm_model.eval()

    print("[RAG] All models loaded and ready.")


# ── Retrieval ─────────────────────────────────────────────────────────────────

def _simple_tokenize(text: str) -> list:
    """Lightweight BM25 tokenizer."""
    return re.findall(r"\b\w+\b", text.lower())


def retrieve(query: str, top_k: int = 3, mode: str = "hybrid") -> list:
    """
    Retrieve the top-k most relevant documents for a query.

    Args:
        query : User's natural language question
        top_k : Number of documents to retrieve
        mode  : "vector" (semantic only), "bm25" (keyword only),
                or "hybrid" (weighted fusion, recommended)

    Returns:
        List of dicts with keys: score, category, intent, instruction, response
    """
    _load_models()

    query_vec = _embed_model.encode(
        [query], normalize_embeddings=True, convert_to_numpy=True
    ).astype("float32")

    if mode == "vector":
        scores, indices = _faiss_index.search(query_vec, top_k)
        pairs = list(zip(scores[0], indices[0]))

    elif mode == "bm25":
        bm25_scores = _bm25.get_scores(_simple_tokenize(query))
        top_indices = np.argsort(bm25_scores)[::-1][:top_k]
        pairs = [(bm25_scores[i], i) for i in top_indices]

    else:  # hybrid
        vec_scores, vec_indices = _faiss_index.search(query_vec, len(_train_texts))
        vec_map = dict(zip(vec_indices[0], vec_scores[0]))

        bm25_scores = _bm25.get_scores(_simple_tokenize(query))
        bm25_max    = bm25_scores.max() if bm25_scores.max() > 0 else 1.0
        bm25_norm   = bm25_scores / bm25_max

        alpha   = 0.6  # 60% vector, 40% keyword — tuned on validation set
        fused   = [alpha * vec_map.get(i, 0.0) + (1 - alpha) * bm25_norm[i]
                   for i in range(len(_train_texts))]
        top_idx = np.argsort(fused)[::-1][:top_k]
        pairs   = [(fused[i], i) for i in top_idx]

    results = []
    for score, idx in pairs:
        row = _lookup_df.iloc[int(idx)]
        results.append({
            "score"      : float(score),
            "category"   : row["category"],
            "intent"     : row["intent"],
            "instruction": row["instruction_clean"],
            "response"   : row["response_clean"],
        })
    return results


# ── Prompt builder ────────────────────────────────────────────────────────────

def _build_prompt(question: str, retrieved_docs: list) -> str:
    """
    Build the prompt fed to the LLM.

    Format:
        System instruction
        --- context block (retrieved docs) ---
        Question: <user question>
        Answer:

    flan-t5 is an instruction-tuned model — it responds well to this
    explicit question/answer structure without needing few-shot examples.

    Args:
        question      : Original user question
        retrieved_docs: Output of retrieve()

    Returns:
        Single string prompt ready for tokenization
    """
    context_blocks = []
    for i, doc in enumerate(retrieved_docs, 1):
        context_blocks.append(
            f"[Document {i}]\n"
            f"Category: {doc['category']} | Intent: {doc['intent']}\n"
            f"Example question: {doc['instruction']}\n"
            f"Example answer: {doc['response']}"
        )
    context = "\n\n".join(context_blocks)

    prompt = (
        "You are a helpful customer support assistant. "
        "Use the context below to answer the customer's question clearly and politely. "
        "If the context does not cover the question, say so honestly.\n\n"
        f"Context:\n{context}\n\n"
        f"Customer question: {question}\n\n"
        "Answer:"
    )
    return prompt


# ── Generation ────────────────────────────────────────────────────────────────

def generate_answer(prompt: str, max_new_tokens: int = 200) -> str:
    """
    Generate an answer from the prompt using flan-t5-base locally.

    Args:
        prompt         : Full prompt string from _build_prompt()
        max_new_tokens : Max tokens to generate (200 ≈ 2-3 sentences)

    Returns:
        Generated answer string
    """
    _load_models()

    inputs = _llm_tokenizer(
        prompt,
        return_tensors="pt",
        max_length=512,
        truncation=True,
    )
    outputs = _llm_model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        num_beams=4,           # beam search for better quality than greedy
        early_stopping=True,
        no_repeat_ngram_size=3, # prevents repetitive phrases
    )
    return _llm_tokenizer.decode(outputs[0], skip_special_tokens=True)


# ── Main RAG function ─────────────────────────────────────────────────────────

def ask(question: str, top_k: int = 3, retrieval_mode: str = "hybrid") -> dict:
    """
    Full RAG pipeline: retrieve context → build prompt → generate answer.

    This is the single function called by both the notebook and the API.

    Args:
        question       : User's natural language question
        top_k          : Number of context documents to retrieve
        retrieval_mode : "vector", "bm25", or "hybrid"

    Returns:
        dict with keys:
            question    : original question
            answer      : generated answer string
            sources     : list of retrieved documents (for transparency / citation)
            top_intent  : most likely intent detected from top retrieved doc
    """
    _load_models()

    # Step 1 — retrieve
    sources = retrieve(question, top_k=top_k, mode=retrieval_mode)

    # Step 2 — build prompt
    prompt = _build_prompt(question, sources)

    # Step 3 — generate
    answer = generate_answer(prompt)

    return {
        "question"   : question,
        "answer"     : answer,
        "sources"    : sources,
        "top_intent" : sources[0]["intent"] if sources else "unknown",
    }


if __name__ == "__main__":
    # Quick smoke test — run: python src/rag_chain.py
    result = ask("I want to cancel my order")
    print(f"\nQ: {result['question']}")
    print(f"A: {result['answer']}")
    print(f"Intent detected: {result['top_intent']}")
