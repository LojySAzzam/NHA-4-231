# RAG Customer Support Chatbot

**NHA-4-231 · DEPI Capstone Project · July 2026**

A production-shaped Retrieval-Augmented Generation (RAG) system that automates Tier-1 customer support. Every answer is grounded in real support transcripts, cited with sources, and monitored via MLflow — built across 5 DEPI milestones.

---

## What It Does

Instead of letting an LLM answer freeform (and hallucinate policies or invent order numbers), this system:

1. Takes a customer query
2. Retrieves the most relevant past Q&A pairs from a vector store using hybrid search (FAISS dense + BM25 keyword, fused via Reciprocal Rank Fusion)
3. Feeds those as grounded context into Llama 3.3 70B (via Groq)
4. Returns a cited, hallucination-resistant answer through a FastAPI backend and React chat UI

---

## Tech Stack

| Layer | Technology |
|---|---|
| Dataset | Bitext Customer Support LLM Dataset (26,872 rows, 27 intents) |
| Embeddings | Google `gemini-embedding-001` (3072-dim) |
| Vector store | FAISS `IndexFlatIP` (cosine similarity) |
| Keyword search | BM25 (`rank-bm25`) |
| Hybrid fusion | Reciprocal Rank Fusion (RRF, k=60) |
| LLM generation | Groq `llama-3.3-70b-versatile` |
| REST API | FastAPI + Uvicorn |
| Frontend | React + Vite + Tailwind + shadcn/ui |
| MLOps | MLflow 3.14 (SQLite backend) |
| Evaluation | BLEU, ROUGE-1/2/L, intent match rate |

---

## Project Structure

```
NHA-4-231/
├── notebooks/
│   ├── Milestone_1_VSCode.ipynb      # Data ingestion, cleaning, EDA, stratified splits
│   ├── Milestone_2_Local.ipynb       # FAISS index, BM25, hybrid search, evaluation
│   ├── Milestone_3_Local.ipynb       # RAG chain, REST API, end-to-end testing
│   └── Milestone_4_Local.ipynb       # MLflow tracking, monitoring dashboard, retraining
├── src/
│   ├── rag_chain.py                  # Core RAG logic (shared by notebook and API)
│   └── api.py                        # FastAPI server
├── rag-frontend/                     # React chat UI
├── data/
│   ├── train_df.csv                  # 17,268-row training corpus
│   ├── val_df.csv                    # 1,252-row validation set
│   ├── test_df.csv                   # 1,251-row evaluation vault
│   ├── noisy_df.csv                  # 2,918-row robustness stress-test set
│   └── faiss_index/
│       ├── train_index.faiss         # FAISS vector index
│       ├── train_embeddings.npy      # Embedding matrix
│       ├── train_lookup.csv          # Row → intent/category mapping
│       ├── bm25_corpus.pkl           # Serialised BM25 index
│       └── embeddings_checkpoint.npy # Resumable rebuild checkpoint
├── reports/
│   └── monitoring_dashboard.png      # 4-panel MLflow monitoring chart
├── mlflow.db                         # SQLite MLflow tracking store
├── .env                              # API keys — never committed (see below)
├── .gitignore
└── requirements.txt
```

---

## Evaluation Results

| Metric | Value | Threshold | Status |
|---|---|---|---|
| Intent match rate | 97.67% | — | ✅ |
| ROUGE-1 | 0.566 | > 0.40 | ✅ |
| ROUGE-2 | 0.279 | — | ✅ |
| ROUGE-L | 0.389 | > 0.35 | ✅ |
| BLEU | 0.213 | — | ✅ |
| Avg live latency | ~1,440 ms | < 5,000 ms | ✅ |

---

## Setup

### Prerequisites

- Python 3.10+
- Node.js 18+ (for the React frontend)
- A Gemini API key — [aistudio.google.com](https://aistudio.google.com) (free)
- A Groq API key — [console.groq.com](https://console.groq.com) (free)

### 1. Clone and create environment

```bash
git clone https://github.com/LojySAzzam/NHA-4-231.git
cd NHA-4-231
python -m venv venv
venv\Scripts\activate       # Windows
# source venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
```

### 2. Configure API keys

Create a `.env` file in the project root (never commit this):

```
GEMINI_API_KEY=AIza...
GROQ_API_KEY=gsk_...
GROQ_MODEL_NAME=llama-3.3-70b-versatile
```

### 3. Run Milestone 1 to generate data files

Open `notebooks/Milestone_1_VSCode.ipynb` in VS Code, select the `venv` kernel, and run all cells. This downloads the dataset and saves `train_df.csv`, `val_df.csv`, `test_df.csv`, and `noisy_df.csv` to `data/`.

### 4. Build the FAISS index

Open `notebooks/Milestone_3_Local.ipynb`, run the `load_all()` cell, then:

```python
rebuild_index()
```

This embeds the training corpus using Gemini (3072-dim vectors) and saves the FAISS index to `data/faiss_index/`. The free tier allows ~1,000 requests/day; the process is checkpoint-resumable — re-running `rebuild_index()` picks up where it left off.

> **Note:** Due to Gemini free-tier quotas, the index may be partially built. The system automatically falls back to BM25-only retrieval when the embedding quota is exhausted, ensuring the chatbot remains fully functional at all times.

### 5. Start the API server

```bash
uvicorn src.api:app --reload --port 8000
```

The server starts at `http://localhost:8000`. Interactive docs available at `http://localhost:8000/docs`.

### 6. Start the React frontend

```bash
cd rag-frontend
npm install
npm run dev
```

Frontend runs at `http://localhost:5173` and connects to the API automatically.

### 7. View the MLflow dashboard

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Open `http://localhost:5000` to see all logged runs, metrics, and charts.

---

## API Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Readiness probe — returns `{status, models_loaded}` |
| `POST` | `/api/chat` | Full RAG query — `{message, top_k, use_hybrid}` → `{answer, sources, retrieval}` |
| `GET` | `/search` | Retrieval-only, no LLM — `?message=&top_k=&use_hybrid=` |
| `GET` | `/docs` | Swagger UI |

**Example request:**

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "I want to cancel my order", "top_k": 3, "use_hybrid": true}'
```

---

## Azure AI Search

The Azure AI Search index schema is fully defined in the codebase. Deployment was blocked by account activation issues across all team accounts, including DEPI-provided accounts. FAISS serves as a functionally identical local substitute. The Azure migration is the immediate next step post-submission.

---

## MLOps

Every `/api/chat` request is automatically logged to MLflow with:

- **Params:** query, top_k, retrieval mode, index size, embedding model, LLM model
- **Metrics:** latency_ms, BLEU, ROUGE-1/2/L, top-1 retrieval score
- **Artifacts:** full result JSON

A rolling 20-run window drives automated retraining triggers:
- ROUGE-1 drops below 0.40
- Average latency exceeds 5,000 ms
- Index completeness below 50%

---

## Team

**NHA-4-231 · Digital Egypt Pioneers Initiative (DEPI)**

- Lojyn Ahmed Azzam
- Marawan Mohamed
- Doha Osama
- Makary Nour
- Shahd Salah Hussein

Repository: [github.com/LojySAzzam/NHA-4-231](https://github.com/LojySAzzam/NHA-4-231)
