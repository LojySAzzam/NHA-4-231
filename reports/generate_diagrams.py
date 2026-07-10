"""
generate_diagrams.py
--------------------
Generates all diagrams needed for the RAG Chatbot technical report:
  1.  Use Case Diagram
  2.  Sequence Diagram (RAG query flow)
  3.  Class Diagram (src/ modules)
  4.  Data Flow Diagram (DFD Level 0 + Level 1)
  5.  Component Diagram
  6.  Activity Diagram
  7.  Gantt Chart
  8.  System Architecture Overview

All saved to ./report_diagrams/
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patches as mpatch
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

OUT = "./report_diagrams"
os.makedirs(OUT, exist_ok=True)

# ── Color palette ─────────────────────────────────────────────────────────────
NAVY   = "#021B3A"
BLUE   = "#065A82"
TEAL   = "#1C7293"
MINT   = "#02C39A"
WHITE  = "#FFFFFF"
LGRAY  = "#E8EFF4"
MGRAY  = "#8BA3B5"
DGRAY  = "#4A6072"

def box(ax, x, y, w, h, text, fc=BLUE, tc=WHITE, fs=10, bold=False, radius=0.02):
    fancy = FancyBboxPatch((x - w/2, y - h/2), w, h,
        boxstyle=f"round,pad=0", facecolor=fc, edgecolor=WHITE, linewidth=1.5)
    ax.add_patch(fancy)
    ax.text(x, y, text, ha="center", va="center", fontsize=fs,
            color=tc, fontweight="bold" if bold else "normal",
            fontfamily="DejaVu Sans", wrap=True,
            multialignment="center")

def arrow(ax, x1, y1, x2, y2, label="", color=MGRAY, lw=1.5):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="->", color=color, lw=lw))
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx+0.05, my, label, fontsize=8, color=DGRAY,
                fontfamily="DejaVu Sans")

def oval(ax, x, y, w, h, text, fc=TEAL, tc=WHITE, fs=10):
    ell = mpatches.Ellipse((x, y), w, h, facecolor=fc, edgecolor=WHITE, linewidth=1.5)
    ax.add_patch(ell)
    ax.text(x, y, text, ha="center", va="center", fontsize=fs,
            color=tc, fontfamily="DejaVu Sans", multialignment="center")

def stick_figure(ax, x, y, label, color=NAVY):
    # Head
    head = plt.Circle((x, y+0.35), 0.1, color=color, zorder=5)
    ax.add_patch(head)
    # Body
    ax.plot([x, x], [y+0.25, y-0.1], color=color, lw=2, zorder=5)
    # Arms
    ax.plot([x-0.2, x+0.2], [y+0.05, y+0.05], color=color, lw=2, zorder=5)
    # Legs
    ax.plot([x, x-0.15], [y-0.1, y-0.4], color=color, lw=2, zorder=5)
    ax.plot([x, x+0.15], [y-0.1, y-0.4], color=color, lw=2, zorder=5)
    ax.text(x, y-0.55, label, ha="center", va="top", fontsize=9,
            color=color, fontfamily="DejaVu Sans", fontweight="bold")

def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {name}")

# ═══════════════════════════════════════════════════════════════
# 1. USE CASE DIAGRAM
# ═══════════════════════════════════════════════════════════════
print("1. Use Case Diagram...")
fig, ax = plt.subplots(figsize=(14, 10), facecolor=LGRAY)
ax.set_xlim(0, 14); ax.set_ylim(0, 10)
ax.axis("off")
ax.set_facecolor(LGRAY)

# Title
ax.text(7, 9.5, "Use Case Diagram — RAG Customer Support Chatbot",
        ha="center", va="center", fontsize=14, fontweight="bold",
        color=NAVY, fontfamily="DejaVu Sans")

# System boundary
sys_box = FancyBboxPatch((2.5, 1.0), 9, 8,
    boxstyle="round,pad=0.1", facecolor=WHITE,
    edgecolor=BLUE, linewidth=2, linestyle="--")
ax.add_patch(sys_box)
ax.text(7, 8.8, "RAG Support Chatbot System", ha="center", fontsize=11,
        color=BLUE, fontweight="bold", fontfamily="DejaVu Sans")

# Actors
stick_figure(ax, 0.8, 6.5, "Customer")
stick_figure(ax, 0.8, 3.0, "Support\nAgent")
stick_figure(ax, 13.2, 7.0, "Gemini\nEmbedding\nAPI", color=TEAL)
stick_figure(ax, 13.2, 4.5, "Groq\nLLM API", color=TEAL)
stick_figure(ax, 13.2, 2.0, "MLflow\nSystem", color=TEAL)

# Use cases (ovals)
use_cases = [
    (7.0, 7.8, "Submit support query"),
    (7.0, 6.6, "Receive generated answer"),
    (7.0, 5.4, "View source documents"),
    (7.0, 4.2, "Search knowledge base"),
    (7.0, 3.0, "Monitor system performance"),
    (7.0, 1.8, "Trigger index retraining"),
]
for x, y, txt in use_cases:
    oval(ax, x, y, 3.8, 0.7, txt, fc=BLUE, fs=9)

# Customer connections
for _, y, _ in use_cases[:4]:
    ax.annotate("", xy=(5.1, y), xytext=(1.3, 6.3 if y > 5 else 3.3),
        arrowprops=dict(arrowstyle="-", color=MGRAY, lw=1))

# Agent connections
ax.annotate("", xy=(5.1, 3.0), xytext=(1.3, 3.0),
    arrowprops=dict(arrowstyle="-", color=MGRAY, lw=1))
ax.annotate("", xy=(5.1, 1.8), xytext=(1.3, 3.0),
    arrowprops=dict(arrowstyle="-", color=MGRAY, lw=1))

# External system connections
ax.annotate("", xy=(8.9, 7.8), xytext=(12.7, 7.0),
    arrowprops=dict(arrowstyle="-", color=TEAL, lw=1, linestyle="dashed"))
ax.annotate("", xy=(8.9, 6.6), xytext=(12.7, 4.5),
    arrowprops=dict(arrowstyle="-", color=TEAL, lw=1, linestyle="dashed"))
ax.annotate("", xy=(8.9, 3.0), xytext=(12.7, 2.0),
    arrowprops=dict(arrowstyle="-", color=TEAL, lw=1, linestyle="dashed"))

save(fig, "01_use_case_diagram.png")

# ═══════════════════════════════════════════════════════════════
# 2. SEQUENCE DIAGRAM
# ═══════════════════════════════════════════════════════════════
print("2. Sequence Diagram...")
fig, ax = plt.subplots(figsize=(16, 11), facecolor=WHITE)
ax.set_xlim(0, 16); ax.set_ylim(0, 11)
ax.axis("off")

ax.text(8, 10.6, "Sequence Diagram — RAG Query Flow",
        ha="center", fontsize=14, fontweight="bold", color=NAVY)

actors = [
    (1.2,  "User /\nFrontend", TEAL),
    (3.8,  "FastAPI\n/ask", BLUE),
    (6.4,  "rag_chain\n.ask()", BLUE),
    (9.0,  "FAISS +\nBM25", NAVY),
    (11.6, "Gemini\nEmbed API", TEAL),
    (14.2, "Groq\nLLM API", TEAL),
]

# Lifeline headers
for x, label, c in actors:
    box(ax, x, 10.1, 1.6, 0.7, label, fc=c, fs=9, bold=True)
    ax.plot([x, x], [9.75, 0.3], color=MGRAY, lw=1, linestyle="--")

# Messages
msgs = [
    (1.2, 3.8, 9.4, "POST /ask {question}"),
    (3.8, 6.4, 9.0, "ask(query, top_k=3)"),
    (6.4, 11.6, 8.6, "embed_content(query)"),
    (11.6, 6.4, 8.2, "← 3072-dim vector"),
    (6.4, 9.0, 7.8, "FAISS.search(vector, n)"),
    (9.0, 6.4, 7.4, "← top-3 doc indices"),
    (6.4, 9.0, 7.0, "BM25.get_scores(tokens)"),
    (9.0, 6.4, 6.6, "← BM25 scores"),
    (6.4, 6.4, 6.2, "fuse_scores(α=0.6)  [self]"),
    (6.4, 6.4, 5.8, "build_prompt(query, docs)  [self]"),
    (6.4, 14.2, 5.4, "chat.completions.create(prompt)"),
    (14.2, 6.4, 5.0, "← generated answer (300 tokens)"),
    (6.4, 3.8, 4.6, "← {answer, sources, latency}"),
    (3.8, 3.8, 4.2, "log_run(mlflow)  [self]"),
    (3.8, 1.2, 3.8, "← JSON response"),
]

for x1, x2, y, label in msgs:
    is_self = x1 == x2
    is_return = x2 < x1 and label.startswith("←")
    color = MINT if is_return else NAVY
    style = "<-" if is_return else "->"
    if is_self:
        ax.annotate("", xy=(x2+0.9, y-0.2), xytext=(x2+0.9, y),
            arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.2))
        ax.plot([x2, x2+0.9, x2+0.9], [y, y, y-0.2], color=BLUE, lw=1.2)
        ax.text(x2+1.0, y-0.1, label, fontsize=8, color=DGRAY, va="center")
    else:
        ax.annotate("", xy=(x2, y), xytext=(x1, y),
            arrowprops=dict(arrowstyle=style, color=color, lw=1.3))
        mx = (x1+x2)/2
        offset = 0.08 if not is_return else -0.15
        ax.text(mx, y+offset, label, ha="center", fontsize=8,
                color=DGRAY if not is_return else TEAL)

# Activation boxes
for x, _, c in actors[1:4]:
    ax.add_patch(FancyBboxPatch((x-0.12, 1.0), 0.24, 8.4,
        boxstyle="square,pad=0", facecolor=c, alpha=0.15, edgecolor=c, lw=0.5))

save(fig, "02_sequence_diagram.png")

# ═══════════════════════════════════════════════════════════════
# 3. CLASS DIAGRAM
# ═══════════════════════════════════════════════════════════════
print("3. Class Diagram...")
fig, ax = plt.subplots(figsize=(16, 11), facecolor=LGRAY)
ax.set_xlim(0, 16); ax.set_ylim(0, 11)
ax.axis("off")
ax.set_facecolor(LGRAY)

ax.text(8, 10.6, "Class Diagram — RAG Chatbot Source Modules",
        ha="center", fontsize=14, fontweight="bold", color=NAVY)

def class_box(ax, x, y, name, attrs, methods, w=3.5):
    total_h = 0.45 + len(attrs)*0.28 + 0.05 + len(methods)*0.28 + 0.15
    # Header
    ax.add_patch(FancyBboxPatch((x, y-total_h), w, total_h,
        boxstyle="square,pad=0", facecolor=WHITE, edgecolor=BLUE, lw=1.5))
    ax.add_patch(FancyBboxPatch((x, y-0.45), w, 0.45,
        boxstyle="square,pad=0", facecolor=BLUE, edgecolor=BLUE, lw=1.5))
    ax.text(x+w/2, y-0.22, name, ha="center", va="center",
            fontsize=10, fontweight="bold", color=WHITE, fontfamily="monospace")
    # Attrs
    cy = y - 0.55
    for a in attrs:
        ax.text(x+0.12, cy, a, fontsize=8, color=NAVY, fontfamily="monospace", va="top")
        cy -= 0.28
    # Divider
    ax.plot([x, x+w], [cy+0.1, cy+0.1], color=BLUE, lw=0.8)
    cy -= 0.05
    for m in methods:
        ax.text(x+0.12, cy, m, fontsize=8, color=DGRAY, fontfamily="monospace", va="top")
        cy -= 0.28
    return y - total_h

# rag_chain module
y1 = class_box(ax, 0.3, 10.2, "rag_chain (module)",
    ["- _faiss_index: faiss.Index",
     "- _train_df: pd.DataFrame",
     "- _bm25: BM25Okapi",
     "- _groq_client: Groq",
     "- EMBEDDING_MODEL: str",
     "- EMBEDDING_DIMS: int"],
    ["+ load_all(index_dir) → None",
     "+ rebuild_index(index_dir) → None",
     "+ ask(query, top_k, use_hybrid) → dict",
     "+ search(query, top_k) → list",
     "- _embed_query(query) → np.ndarray",
     "- _retrieve_hybrid(query, top_k) → list",
     "- _retrieve_vector(query, top_k) → list",
     "- _build_prompt(query, docs) → str",
     "- _generate(prompt) → str"], w=5.2)

# api module
y2 = class_box(ax, 6.2, 10.2, "api (FastAPI app)",
    ["- app: FastAPI",
     "- BASE_URL: str = localhost:8000"],
    ["+ health_check() → HealthResponse",
     "+ ask_question(req) → AskResponse",
     "+ search_docs(query) → list[SourceDoc]",
     "+ lifespan() → AsyncContextManager"], w=4.8)

# Pydantic models
y3 = class_box(ax, 6.2, 5.5, "AskRequest (Pydantic)",
    ["+ question: str",
     "+ top_k: int = 3",
     "+ use_hybrid: bool = True"], [], w=4.0)

y4 = class_box(ax, 11.2, 5.5, "AskResponse (Pydantic)",
    ["+ query: str",
     "+ answer: str",
     "+ sources: list[SourceDoc]",
     "+ retrieval: str"], [], w=4.5)

y5 = class_box(ax, 0.3, 5.5, "SourceDoc (Pydantic)",
    ["+ score: float",
     "+ category: str",
     "+ intent: str",
     "+ instruction: str",
     "+ response: str"], [], w=4.5)

# Relationships
# api uses rag_chain
ax.annotate("", xy=(5.5, 8.5), xytext=(6.2, 8.5),
    arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.5))
ax.text(5.75, 8.65, "uses", fontsize=8, color=TEAL, ha="center")

# api uses AskRequest
ax.annotate("", xy=(8.6, 5.5), xytext=(8.6, 6.3),
    arrowprops=dict(arrowstyle="->", color=MGRAY, lw=1.2))
ax.text(8.7, 5.9, "receives", fontsize=8, color=MGRAY)

# api returns AskResponse
ax.annotate("", xy=(11.4, 5.5), xytext=(10.2, 6.3),
    arrowprops=dict(arrowstyle="->", color=MGRAY, lw=1.2))
ax.text(10.5, 5.85, "returns", fontsize=8, color=MGRAY)

# SourceDoc in AskResponse
ax.annotate("", xy=(4.8, 4.2), xytext=(11.2, 4.5),
    arrowprops=dict(arrowstyle="->", color=NAVY, lw=1.2, linestyle="dashed"))
ax.text(7.5, 4.45, "contains list of", fontsize=8, color=NAVY, ha="center")

save(fig, "03_class_diagram.png")

# ═══════════════════════════════════════════════════════════════
# 4. DATA FLOW DIAGRAM — Level 0 (Context)
# ═══════════════════════════════════════════════════════════════
print("4a. DFD Level 0...")
fig, ax = plt.subplots(figsize=(12, 8), facecolor=WHITE)
ax.set_xlim(0, 12); ax.set_ylim(0, 8)
ax.axis("off")

ax.text(6, 7.6, "DFD Level 0 (Context Diagram) — RAG Support Chatbot",
        ha="center", fontsize=13, fontweight="bold", color=NAVY)

# Central process
oval(ax, 6, 4, 3.5, 1.8, "RAG Customer\nSupport Chatbot\nSystem", fc=BLUE, fs=11)

# External entities
entities = [
    (1.2, 6.5, "Customer"),
    (10.8, 6.5, "Gemini\nEmbedding API"),
    (1.2, 1.5, "Support Agent /\nAdmin"),
    (10.8, 1.5, "Groq\nLLM API"),
    (6.0, 0.6, "MLflow\nTracking"),
]
for x, y, label in entities:
    box(ax, x, y, 2.0, 0.9, label, fc=NAVY, fs=9)

# Flows
flows = [
    (2.2, 6.5, 4.25, 5.0, "Support query"),
    (4.25, 4.5, 2.2, 6.2, "Generated answer"),
    (7.75, 5.0, 9.8, 6.5, "Query text"),
    (9.8, 6.2, 7.75, 4.5, "3072-dim vector"),
    (7.75, 3.5, 9.8, 1.8, "RAG prompt"),
    (9.8, 1.5, 7.75, 3.8, "Generated text"),
    (2.2, 1.5, 4.25, 3.2, "Monitor / retrain"),
    (4.25, 3.2, 2.2, 1.8, "MLflow reports"),
    (6.0, 2.9, 6.0, 1.1, "Experiment logs"),
]
for x1, y1, x2, y2, label in flows:
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.5))
    mx, my = (x1+x2)/2, (y1+y2)/2
    ax.text(mx+0.1, my+0.1, label, fontsize=8, color=DGRAY, ha="center")

save(fig, "04a_dfd_level0.png")

# ═══════════════════════════════════════════════════════════════
# 4b. DATA FLOW DIAGRAM — Level 1
# ═══════════════════════════════════════════════════════════════
print("4b. DFD Level 1...")
fig, ax = plt.subplots(figsize=(16, 10), facecolor=WHITE)
ax.set_xlim(0, 16); ax.set_ylim(0, 10)
ax.axis("off")

ax.text(8, 9.6, "DFD Level 1 — Internal RAG Pipeline",
        ha="center", fontsize=13, fontweight="bold", color=NAVY)

# Processes (circles/ovals)
processes = [
    (2.0, 7.5, "P1\nEmbed\nQuery"),
    (5.5, 7.5, "P2\nVector\nSearch"),
    (5.5, 4.5, "P3\nBM25\nSearch"),
    (8.5, 6.0, "P4\nFuse\nScores"),
    (11.5, 6.0, "P5\nBuild\nPrompt"),
    (14.5, 6.0, "P6\nGenerate\nAnswer"),
]
for x, y, label in processes:
    oval(ax, x, y, 2.2, 1.2, label, fc=BLUE, fs=9)

# Data stores
stores = [
    (5.5, 9.2, "DS1: FAISS Index"),
    (5.5, 2.8, "DS2: BM25 Corpus"),
    (11.5, 9.2, "DS3: Bitext Train Lookup"),
    (2.0, 9.2, "DS4: Gemini API"),
    (14.5, 9.2, "DS5: Groq API"),
    (8.5, 9.2, "DS6: MLflow DB"),
]
for x, y, label in stores:
    ax.add_patch(FancyBboxPatch((x-1.8, y-0.25), 3.6, 0.5,
        boxstyle="square,pad=0", facecolor=LGRAY, edgecolor=NAVY, lw=1.2))
    ax.text(x, y, label, ha="center", va="center", fontsize=8,
            color=NAVY, fontfamily="DejaVu Sans")

# External entities
box(ax, 0.7, 5.5, 1.2, 0.6, "Customer", fc=NAVY, fs=9)
box(ax, 0.7, 1.5, 1.2, 0.6, "Agent", fc=NAVY, fs=9)

# Flow arrows
flow_data = [
    (0.7, 5.5, 2.0, 7.2, "query text"),
    (2.0, 8.9, 2.0, 8.1, ""),         # DS4
    (2.0, 8.1, 2.0, 8.1, ""),
    (3.1, 7.5, 4.4, 7.5, "query vector"),
    (5.5, 8.9, 5.5, 8.1, ""),         # DS1
    (6.6, 7.5, 7.4, 6.3, "vec scores"),
    (5.5, 3.4, 5.5, 3.9, ""),         # DS2
    (6.6, 4.5, 7.4, 5.7, "BM25 scores"),
    (9.6, 6.0, 10.4, 6.0, "top-3 indices"),
    (11.5, 8.9, 11.5, 8.3, ""),       # DS3
    (11.5, 8.3, 11.5, 6.6, "doc texts"),
    (12.6, 6.0, 13.4, 6.0, "prompt"),
    (14.5, 8.9, 14.5, 6.6, ""),       # DS5
    (14.5, 5.4, 14.5, 1.8, "answer"),
    (8.5, 8.9, 8.5, 6.6, ""),         # DS6
]

simple_arrows = [
    (1.3, 5.5, 1.8, 7.2, "query text"),
    (2.0, 8.9, 2.0, 8.15, "call API"),
    (3.1, 7.5, 4.4, 7.5, "query vector"),
    (5.5, 8.9, 5.5, 8.15, "read index"),
    (6.6, 7.5, 7.4, 6.4, "vec scores"),
    (5.5, 3.4, 5.5, 3.95, "read corpus"),
    (6.6, 4.5, 7.4, 5.6, "BM25 scores"),
    (9.6, 6.0, 10.4, 6.0, "top-3 idx"),
    (11.5, 8.9, 11.5, 8.25, "lookup docs"),
    (11.5, 8.25, 11.5, 6.6, "doc texts"),
    (12.6, 6.0, 13.4, 6.0, "prompt"),
    (14.5, 8.9, 14.5, 6.6, "call Groq"),
    (14.5, 5.4, 14.5, 1.8, "answer"),
    (8.5, 8.9, 8.5, 6.6, "log metrics"),
]
for x1, y1, x2, y2, label in simple_arrows:
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.3))
    mx, my = (x1+x2)/2, (y1+y2)/2
    if label:
        ax.text(mx+0.1, my+0.1, label, fontsize=7.5, color=DGRAY)

save(fig, "04b_dfd_level1.png")

# ═══════════════════════════════════════════════════════════════
# 5. COMPONENT DIAGRAM
# ═══════════════════════════════════════════════════════════════
print("5. Component Diagram...")
fig, ax = plt.subplots(figsize=(16, 10), facecolor=LGRAY)
ax.set_xlim(0, 16); ax.set_ylim(0, 10)
ax.axis("off")
ax.set_facecolor(LGRAY)

ax.text(8, 9.6, "Component Diagram — System Architecture",
        ha="center", fontsize=14, fontweight="bold", color=NAVY)

# Layer labels
for y, label, c in [(8.8, "PRESENTATION LAYER", TEAL),
                     (6.3, "APPLICATION LAYER", BLUE),
                     (3.8, "DATA LAYER", NAVY),
                     (1.3, "EXTERNAL SERVICES", MGRAY)]:
    ax.add_patch(FancyBboxPatch((0.2, y-0.9), 0.35, 0.7,
        boxstyle="square,pad=0", facecolor=c, edgecolor=c))
    ax.text(0.45, y-0.55, label, fontsize=7, color=c, fontweight="bold",
            rotation=90, va="center", ha="center")

# Components per layer
components = {
    "presentation": [(3, 8.3, "React\nFrontend", TEAL),
                     (7, 8.3, "Swagger\nUI /docs", TEAL),
                     (11, 8.3, "MLflow\nUI :5000", TEAL)],
    "application":  [(3, 6.0, "FastAPI\nREST API\n:8000", BLUE),
                     (7, 6.0, "rag_chain\n.py", BLUE),
                     (11, 6.0, "MLflow\nTracking", BLUE),
                     (14, 6.0, "api.py\n(router)", BLUE)],
    "data":         [(3, 3.5, "FAISS\nIndexFlatIP\n(local)", NAVY),
                     (7, 3.5, "BM25\nCorpus\n(pkl)", NAVY),
                     (11, 3.5, "Train\nLookup\n(csv)", NAVY),
                     (14, 3.5, "mlflow.db\n(SQLite)", NAVY)],
    "external":     [(3, 1.2, "Gemini\nembedding-001", MGRAY),
                     (7, 1.2, "Groq\nLlama 3.3 70B", MGRAY),
                     (11, 1.2, "HuggingFace\nBitext Dataset", MGRAY)],
}

boxes = {}
for layer, comps in components.items():
    for x, y, label, c in comps:
        box(ax, x, y, 2.2, 1.1, label, fc=c, fs=9, bold=True)
        boxes[label.split("\n")[0]] = (x, y)

# Connections
connections = [
    ("React", "FastAPI", "HTTP /ask"),
    ("Swagger", "FastAPI", "HTTP"),
    ("FastAPI", "rag_chain", "calls"),
    ("FastAPI", "api.py", "routes"),
    ("rag_chain", "FAISS", "search()"),
    ("rag_chain", "BM25", "get_scores()"),
    ("rag_chain", "Train", "iloc[idx]"),
    ("rag_chain", "Gemini", "embed_content()"),
    ("rag_chain", "Groq", "generate()"),
    ("MLflow", "mlflow.db", "writes"),
]
for src, dst, label in connections:
    if src in boxes and dst in boxes:
        x1, y1 = boxes[src]
        x2, y2 = boxes[dst]
        ax.annotate("", xy=(x2, y2+0.55), xytext=(x1, y1-0.55),
            arrowprops=dict(arrowstyle="->", color=MINT, lw=1.2,
                           connectionstyle="arc3,rad=0.05"))

save(fig, "05_component_diagram.png")

# ═══════════════════════════════════════════════════════════════
# 6. ACTIVITY DIAGRAM
# ═══════════════════════════════════════════════════════════════
print("6. Activity Diagram...")
fig, ax = plt.subplots(figsize=(10, 14), facecolor=WHITE)
ax.set_xlim(0, 10); ax.set_ylim(0, 14)
ax.axis("off")

ax.text(5, 13.6, "Activity Diagram — RAG Query Processing",
        ha="center", fontsize=13, fontweight="bold", color=NAVY)

activities = [
    (5.0, 12.8, "START", "circle", NAVY),
    (5.0, 11.8, "Receive customer query\nvia POST /ask", "rect", BLUE),
    (5.0, 10.5, "Embed query with\nGemini embedding-001", "rect", BLUE),
    (5.0, 9.3,  "Embedding quota\navailable?", "diamond", TEAL),
    (2.0, 8.0,  "Pure BM25\nkeyword retrieval", "rect", MGRAY),
    (7.5, 8.0,  "FAISS vector search\n+ BM25 hybrid fusion", "rect", BLUE),
    (5.0, 6.8,  "Retrieve top-3\ndocuments", "rect", BLUE),
    (5.0, 5.6,  "Build prompt with\ncontext documents", "rect", BLUE),
    (5.0, 4.4,  "Generate answer via\nGroq Llama 3.3 70B", "rect", BLUE),
    (5.0, 3.2,  "Log run to\nMLflow", "rect", NAVY),
    (5.0, 2.0,  "Return JSON response\n(answer + sources)", "rect", TEAL),
    (5.0, 1.0,  "END", "circle", NAVY),
]

for x, y, label, shape, c in activities:
    if shape == "circle":
        circ = plt.Circle((x, y), 0.35, color=c, zorder=5)
        ax.add_patch(circ)
        if label == "START":
            ax.text(x, y, "●", ha="center", va="center",
                    fontsize=18, color=WHITE, zorder=6)
        else:
            inner = plt.Circle((x, y), 0.22, color=WHITE, zorder=6)
            ax.add_patch(inner)
            circ2 = plt.Circle((x, y), 0.35, color=c,
                fill=False, lw=2.5, zorder=7)
            ax.add_patch(circ2)
    elif shape == "diamond":
        d = 0.7
        diamond = plt.Polygon(
            [(x, y+d*0.6), (x+d, y), (x, y-d*0.6), (x-d, y)],
            facecolor=c, edgecolor=WHITE, lw=1.5, zorder=5)
        ax.add_patch(diamond)
        ax.text(x, y, label, ha="center", va="center",
                fontsize=8, color=WHITE, fontweight="bold", zorder=6)
    else:
        box(ax, x, y, 3.4, 0.7, label, fc=c, fs=9)

# Main flow arrows
main_flow = [(5,12.45), (5,12.15), (5,11.15), (5,10.85),
             (5,9.65), (5,9.0)]
for i in range(len(main_flow)-1):
    x1,y1 = main_flow[i]; x2,y2 = main_flow[i+1]
    ax.annotate("", xy=(x2,y2), xytext=(x1,y1),
        arrowprops=dict(arrowstyle="->", color=NAVY, lw=1.5))

# Decision branches
ax.annotate("", xy=(2.0, 8.35), xytext=(4.3, 9.0),
    arrowprops=dict(arrowstyle="->", color=MGRAY, lw=1.3))
ax.text(3.0, 8.85, "No", fontsize=9, color=MGRAY, fontweight="bold")

ax.annotate("", xy=(7.5, 8.35), xytext=(5.7, 9.0),
    arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.3))
ax.text(6.8, 8.85, "Yes", fontsize=9, color=TEAL, fontweight="bold")

# Merge back
ax.annotate("", xy=(5.0, 7.15), xytext=(2.0, 7.65),
    arrowprops=dict(arrowstyle="->", color=MGRAY, lw=1.3))
ax.annotate("", xy=(5.0, 7.15), xytext=(7.5, 7.65),
    arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.3))

# Lower flow
lower = [(5,6.45),(5,5.95),(5,5.25),(5,4.75),
         (5,4.05),(5,3.55),(5,2.65),(5,2.35),(5,1.35)]
for i in range(len(lower)-1):
    x1,y1 = lower[i]; x2,y2 = lower[i+1]
    ax.annotate("", xy=(x2,y2), xytext=(x1,y1),
        arrowprops=dict(arrowstyle="->", color=NAVY, lw=1.5))

save(fig, "06_activity_diagram.png")

# ═══════════════════════════════════════════════════════════════
# 7. GANTT CHART
# ═══════════════════════════════════════════════════════════════
print("7. Gantt Chart...")
fig, ax = plt.subplots(figsize=(16, 10), facecolor=WHITE)
ax.set_facecolor(WHITE)

tasks = [
    ("M1: Data collection & ingestion",      1, 1, TEAL),
    ("M1: PII cleaning & preprocessing",     1, 1, TEAL),
    ("M1: EDA report",                       2, 1, TEAL),
    ("M2: Embedding generation (partial)",   2, 3, BLUE),
    ("M2: FAISS index + BM25 corpus",        3, 1, BLUE),
    ("M2: Retrieval evaluation (BLEU/ROUGE)", 4, 1, BLUE),
    ("M3: RAG chain development",            4, 1, NAVY),
    ("M3: FastAPI REST API",                 5, 1, NAVY),
    ("M3: React frontend integration",       5, 1, NAVY),
    ("M3: Azure deployment (blocked)",       6, 1, MGRAY),
    ("M4: MLflow tracking setup",            6, 1, "#8B5CF6"),
    ("M4: Monitoring dashboard",             7, 1, "#8B5CF6"),
    ("M4: Retraining pipeline",              7, 1, "#8B5CF6"),
    ("M5: Technical report",                 8, 1, MINT),
    ("M5: Presentation slides",              8, 1, MINT),
    ("M5: UML diagrams & documentation",     8, 1, MINT),
]

yticks = []
ylabels = []
for i, (label, start, duration, color) in enumerate(reversed(tasks)):
    y = i
    ax.barh(y, duration, left=start-1, height=0.6,
            color=color, edgecolor=WHITE, linewidth=0.5, alpha=0.85)
    yticks.append(y)
    ylabels.append(label)

ax.set_yticks(yticks)
ax.set_yticklabels(ylabels, fontsize=10, fontfamily="DejaVu Sans")
ax.set_xticks(range(9))
ax.set_xticklabels(["", "Week 1-2", "Week 3", "Week 4",
                     "Week 5", "Week 6", "Week 7", "Week 8", "Week 9"],
                   fontsize=10)
ax.set_xlim(0, 8.5)
ax.set_xlabel("Project Timeline", fontsize=11, color=NAVY)
ax.set_title("Project Gantt Chart — RAG Customer Support Chatbot",
             fontsize=14, fontweight="bold", color=NAVY, pad=15)
ax.grid(axis="x", alpha=0.3, color=MGRAY)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Legend
legend_items = [
    mpatches.Patch(color=TEAL, label="Milestone 1"),
    mpatches.Patch(color=BLUE, label="Milestone 2"),
    mpatches.Patch(color=NAVY, label="Milestone 3"),
    mpatches.Patch(color="#8B5CF6", label="Milestone 4"),
    mpatches.Patch(color=MINT, label="Milestone 5"),
    mpatches.Patch(color=MGRAY, label="Blocked (Azure)"),
]
ax.legend(handles=legend_items, loc="lower right", fontsize=9,
          framealpha=0.9, edgecolor=LGRAY)

save(fig, "07_gantt_chart.png")

# ═══════════════════════════════════════════════════════════════
# 8. SYSTEM ARCHITECTURE OVERVIEW
# ═══════════════════════════════════════════════════════════════
print("8. System Architecture Overview...")
fig, ax = plt.subplots(figsize=(16, 10), facecolor=NAVY)
ax.set_xlim(0, 16); ax.set_ylim(0, 10)
ax.axis("off")
ax.set_facecolor(NAVY)

ax.text(8, 9.6, "System Architecture — RAG Customer Support Chatbot",
        ha="center", fontsize=14, fontweight="bold", color=WHITE)

# Pipeline steps with large arrows
steps = [
    (1.5, 5.0, "①\nUser\nQuery", TEAL),
    (4.0, 5.0, "②\nGemini\nEmbedding", BLUE),
    (6.5, 5.0, "③\nFAISS\n+ BM25\nRetrieval", BLUE),
    (9.2, 5.0, "④\nContext\nFusion &\nPrompt Build", BLUE),
    (12.0, 5.0, "⑤\nGroq\nLlama 3.3\nGeneration", BLUE),
    (14.5, 5.0, "⑥\nJSON\nResponse", TEAL),
]
for x, y, label, c in steps:
    box(ax, x, y, 2.2, 2.2, label, fc=c, fs=10, bold=True)

# Arrows between steps
for i in range(len(steps)-1):
    x1 = steps[i][0] + 1.1
    x2 = steps[i+1][0] - 1.1
    ax.annotate("", xy=(x2, 5.0), xytext=(x1, 5.0),
        arrowprops=dict(arrowstyle="->", color=MINT, lw=2.5))

# Supporting components below
supports = [
    (3.5, 2.0, "BM25\nCorpus\n17K rows", MGRAY),
    (6.5, 2.0, "FAISS\nIndex\n(3072-dim)", MGRAY),
    (9.5, 2.0, "Train\nLookup\nCSV", MGRAY),
    (12.5, 2.0, "MLflow\nTracking\nSQLite", MGRAY),
]
for x, y, label, c in supports:
    box(ax, x, y, 2.2, 1.4, label, fc=c, fs=9)
    ax.plot([x, x], [2.7, 3.9], color=MGRAY, lw=1, linestyle=":")

# Evaluation metrics bar
ax.add_patch(FancyBboxPatch((0.5, 0.2), 15, 0.9,
    boxstyle="round,pad=0.05", facecolor=BLUE, edgecolor=TEAL, lw=1))
metrics = "ROUGE-1: 0.566   ·   ROUGE-L: 0.389   ·   BLEU: 0.213   ·   Avg Latency: 1,440ms   ·   Intent Match@5: ~95%   ·   32 MLflow runs"
ax.text(8, 0.65, metrics, ha="center", va="center",
        fontsize=10, color=WHITE, fontfamily="DejaVu Sans")

save(fig, "08_system_architecture.png")

print("\n✓ All 8 diagrams saved to:", OUT)
print("\nFiles:")
for f in sorted(os.listdir(OUT)):
    size = os.path.getsize(os.path.join(OUT, f))
    print(f"  {f}  ({size//1024} KB)")
