"""
Query pipeline — HYBRID retrieval (BM25 keyword search + FAISS vector search)
with optional metadata filtering by source file.

Why hybrid: pure vector search struggles when a short, information-dense
chunk (e.g. "CGPA: 8.0") competes against a long, keyword-heavy document
(e.g. a grades table) for a vague query. BM25 scores on exact term
overlap regardless of chunk length, so combining both catches cases either
one would miss alone.

Run (searches all documents):
    python query.py "What is my CGPA?"

Run (searches only one file):
    python query.py "What are my courses?" --source "resume with work.pdf"
"""

import warnings
warnings.filterwarnings("ignore")

import sys
import os
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["HF_HUB_OFFLINE"] = "1"
import time
import re
import groq
import pickle
import types
import argparse
from dotenv import load_dotenv

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Compatibility shim for langchain_core 1.x with langchain 0.3.x
if "langchain_core.memory" not in sys.modules:
    shim = types.ModuleType("langchain_core.memory")
    shim.BaseMemory = object
    sys.modules["langchain_core.memory"] = shim

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

import config

PROMPT_TEMPLATE = """You are a helpful assistant answering questions using
ONLY the context provided below.

Rules:
- Answer the question based strictly on the provided context.
- If the answer isn't in the context, say "I don't have enough information to answer that" — do not make things up.
- Be factual, concise, and direct.

Context:
{context}

Question: {question}

Answer:"""


def load_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.load_local(
        config.VECTOR_STORE_DIR,
        embeddings,
        allow_dangerous_deserialization=True,
    )


_cached_chunks = None
_cached_hybrid_retrievers = {}

def load_chunks():
    """Load the raw chunk list pickled during ingestion (needed for BM25)."""
    global _cached_chunks
    if _cached_chunks is not None:
        return _cached_chunks
    chunks_path = os.path.join(config.VECTOR_STORE_DIR, "chunks.pkl")
    if not os.path.exists(chunks_path):
        raise FileNotFoundError(
            f"'{chunks_path}' not found. Re-run ingestion (python ingest.py) "
            "to pickle chunks for hybrid search."
        )
    with open(chunks_path, "rb") as f:
        _cached_chunks = pickle.load(f)
    return _cached_chunks


def build_hybrid_retriever(vector_store, k=config.TOP_K):
    """
    Combine BM25 (keyword) and FAISS (vector/semantic) retrieval.

    EnsembleRetriever merges results using Reciprocal Rank Fusion — each
    retriever's ranking (not raw score) contributes, weighted. This means
    a chunk that ranks highly on EITHER exact keyword match OR semantic
    similarity can surface, instead of needing to win on both.
    """
    if k in _cached_hybrid_retrievers:
        return _cached_hybrid_retrievers[k]

    chunks = load_chunks()
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = k

    faiss_retriever = vector_store.as_retriever(search_kwargs={"k": k})

    retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[0.4, 0.6],
    )
    _cached_hybrid_retrievers[k] = retriever
    return retriever


def retrieve(vector_store, question, k=config.TOP_K, source_filter=None):
    """
    Retrieve top-k chunks using hybrid (BM25 + vector) search.
    If source_filter is given, only chunks whose metadata 'source' contains
    that string are eligible.
    """
    fetch_k = k * 5 if source_filter else k
    hybrid_retriever = build_hybrid_retriever(vector_store, k=fetch_k)
    results = hybrid_retriever.invoke(question)

    if source_filter:
        results = [
            doc for doc in results
            if source_filter.lower() in doc.metadata.get("source", "").lower()
        ]

    results = results[:k]

    for i, doc in enumerate(results, 1):
        source = doc.metadata.get("source", "unknown")
        print(f"  [{i}] {source}", flush=True)

    return results


def generate_answer(question, retrieved_docs):
    if not retrieved_docs:
        return "I don't have enough information to answer that."

    context = "\n\n---\n\n".join(doc.page_content for doc in retrieved_docs)
    prompt = PROMPT_TEMPLATE.replace("{context}", context).replace("{question}", question)

    api_key = os.getenv("GROQ_API_KEY")
    model_name = getattr(config, "GROQ_MODEL", "openai/gpt-oss-20b")
    client = groq.Groq(api_key=api_key, timeout=20.0)

    max_retries = 6
    for attempt in range(max_retries):
        try:
            res = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            return res.choices[0].message.content
        except Exception as e:
            err_str = str(e)
            is_ratelimit = "RateLimitError" in type(e).__name__ or "429" in err_str or "rate_limit" in err_str.lower()
            is_conn_err = any(w in err_str.lower() for w in ["connection", "connecterror", "getaddrinfo", "timeout", "remotedisconnected"]) or "Connection" in type(e).__name__
            if is_ratelimit or is_conn_err:
                wait = 2.5 * (attempt + 1)
                m = re.search(r"try again in ([0-9.]+)s", err_str)
                if m:
                    wait = float(m.group(1)) + 1.0
                time.sleep(wait)
            else:
                raise e
    return "I don't have enough information to answer that."


def answer_question(question, source_filter=None):
    vector_store = load_vector_store()
    label = f" (filtered to '{source_filter}')" if source_filter else ""
    print(f"\nRetrieving top {config.TOP_K} chunks (hybrid){label}...\n", flush=True)
    docs = retrieve(vector_store, question, source_filter=source_filter)
    answer = generate_answer(question, docs)
    print(f"\nAnswer:\n{answer}\n", flush=True)
    return answer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("question", help="Your question")
    parser.add_argument(
        "--source", default=None,
        help="Only search chunks from this file, e.g. 'resume with work.pdf'"
    )
    args = parser.parse_args()

    answer_question(args.question, source_filter=args.source)