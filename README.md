# RAG Starter (Phase 1)

A minimal, working retrieval-augmented generation pipeline: drop documents
in, ask questions, get grounded answers with citations to source chunks.

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file in this folder:

```
OPENAI_API_KEY=sk-...
```

No OpenAI key? Swap `OpenAIEmbeddings`/`ChatOpenAI` in `ingest.py`/`query.py`
for `HuggingFaceEmbeddings` (`sentence-transformers/all-MiniLM-L6-v2`) and
`ChatOllama` (any local model via Ollama) — same interface, no API cost.

## Usage

1. Put PDFs or `.txt` files into `documents/`.
2. Build the index:
   ```bash
   python ingest.py
   ```
3. Ask questions:
   ```bash
   python query.py "What does the document say about X?"
   ```

## Design decisions (for interview talking points)

| Decision | Why |
|---|---|
| `RecursiveCharacterTextSplitter`, 800/120 chunk/overlap | Splits on paragraph → sentence → word boundaries first, so chunks stay coherent; overlap prevents losing meaning at chunk edges |
| FAISS (local, file-based) | No server to run for a Phase 1 prototype; swappable for Qdrant/Pinecone later without changing the retrieval interface |
| `text-embedding-3-small` | Strong quality-to-cost ratio; easy to swap for a local model |
| `similarity_search_with_score` instead of plain `similarity_search` | Lets you see and reason about retrieval confidence, not just accept whatever comes back |
| Prompt explicitly says "don't make things up" | Reduces (doesn't eliminate) hallucination when retrieved context doesn't actually answer the question |
| `temperature=0` | Deterministic, grounded answers — you want retrieval quality tested, not creative variation |

## Where to go next (see roadmap Phases 2–5)

- Add hybrid search (BM25 + vector) and a reranker
- Build a small eval set and measure retrieval precision/recall
- Move FAISS → Qdrant for a "real" deployment story
- Add citations back to the user showing which chunk backed which claim
