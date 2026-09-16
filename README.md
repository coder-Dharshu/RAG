# Table-Aware RAG Pipeline

A robust, production-ready Retrieval-Augmented Generation (RAG) pipeline designed to index documents (PDFs, text files), parse tables, store embeddings locally, and generate grounded answers with precise source citations.

---

## Key Features

- **Table-Aware PDF & Text Parsing**: Converts tables in PDFs into structured Markdown representations to preserve context for LLMs.
- **Local & Fast Embeddings**: Uses HuggingFace `sentence-transformers/all-MiniLM-L6-v2` locally or swappable OpenAI embeddings.
- **Fast Vector Search**: FAISS vector store for fast similarity search and retrieval.
- **Groq & LLM Support**: Fast inference via Groq API (or OpenAI models) with deterministic output (`temperature=0`).
- **Source Citation & Chunk Debugging**: Includes CLI tools to test chunking quality and trace answers back to source context.

---

## Setup & Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/coder-Dharshu/RAG.git
   cd RAG
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables**:
   Copy `.env.example` to `.env` and fill in your API key:
   ```bash
   cp .env.example .env
   ```
   Add your `GROQ_API_KEY` (or `OPENAI_API_KEY`):
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ```

---

## Usage

### 1. Ingest Documents
Place your PDFs or `.txt` files into the `documents/` folder, then run:
```bash
python ingest.py
```
This parses the documents, chunks the text into overlapping segments, generates embeddings, and saves the vector store in `vector_store/`.

### 2. Query the RAG Pipeline
Run queries from the command line:
```bash
python query.py "What does the document say about X?"
```

### 3. Debug Chunks (Optional)
To inspect chunk split quality and retrieval candidates:
```bash
python debug_chunks.py
```

---

## Architecture & Design Decisions

| Decision | Implementation Rationale |
|---|---|
| **Chunking Strategy** | `RecursiveCharacterTextSplitter` (300 chars, 50 overlap). Splits on paragraph and sentence boundaries to keep contexts coherent without losing edge details. |
| **Vector Indexing** | FAISS (local, file-based index). Fast, zero-overhead similarity search with easy upgrade paths to Qdrant or Pinecone. |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2`. High-performance local embedding model requiring no extra API cost. |
| **Retriever Scoring** | `similarity_search_with_score`. Evaluates retrieval confidence scores for strict grounding. |
| **Grounded Generation** | System prompt instructs model to answer strictly based on retrieved context, reducing hallucinations. |

---

## Future Enhancements

- Hybrid Search (BM25 + Vector Search) & Cross-Encoder Reranking
- Multi-document metadata filtering
- Web UI / Chat Interface integration
