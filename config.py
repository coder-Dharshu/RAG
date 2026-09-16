"""
Central config so every design decision (chunk size, model, k) lives in one
place — makes it easy to explain trade-offs in an interview and to run
experiments by changing one number.
"""

import os
from dotenv import load_dotenv

os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"

load_dotenv()


# --- Paths ---
DOCS_DIR = "documents"          # drop your PDFs/txt files here
VECTOR_STORE_DIR = "vector_store"

# --- Chunking ---
# Smaller chunks = more precise retrieval but less context per chunk.
# Larger chunks = more context but noisier retrieval (irrelevant text
# riding along with the relevant part).
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50   # ~15% overlap prevents losing meaning at chunk edges

# --- Embeddings ---
# Options:
#   - "text-embedding-3-small" (OpenAI, requires API key, ~$0.02 per 1M tokens)
#   - "sentence-transformers/all-MiniLM-L6-v2" (free, runs locally)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
USE_LOCAL_EMBEDDINGS = True

# --- Retrieval ---
TOP_K = 8   # how many chunks to retrieve per query


# --- Generation ---
# Options:
#   - "groq" (using Groq API, ultra-fast inference)
#   - "openai" (OpenAI, requires OPENAI_API_KEY)
#   - "local" (returns retrieved context chunks directly)
LLM_PROVIDER = "groq"
GROQ_MODEL = "openai/gpt-oss-20b"   # also supports "qwen/qwen3.8-27b"
LLM_MODEL = "gpt-4o-mini"
TEMPERATURE = 0

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


