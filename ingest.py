"""
Phase 1 — Ingestion pipeline.

Loads every PDF/txt file in DOCS_DIR, splits into overlapping chunks,
embeds them, and persists a FAISS index to disk.

Run this once whenever your document set changes:
    python ingest.py
"""

import os
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    DirectoryLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

import config


import glob
from table_aware_loader import build_table_aware_documents


def load_documents():
    """Load all PDFs using table-aware loader and .txt files from DOCS_DIR."""
    docs = []

    pdf_files = glob.glob(os.path.join(config.DOCS_DIR, "**/*.pdf"), recursive=True)
    for pdf_path in pdf_files:
        try:
            pdf_docs = build_table_aware_documents(pdf_path)
            docs.extend(pdf_docs)
            print(f"Loaded {len(pdf_docs)} structured documents from '{os.path.basename(pdf_path)}'.")
        except Exception as e:
            print(f"Fallback to PyPDFLoader for '{pdf_path}': {e}")
            loader = PyPDFLoader(pdf_path)
            docs.extend(loader.load())

    txt_loader = DirectoryLoader(
        config.DOCS_DIR, glob="**/*.txt", loader_cls=TextLoader
    )
    docs.extend(txt_loader.load())

    if not docs:
        raise FileNotFoundError(
            f"No .pdf or .txt files found in '{config.DOCS_DIR}/'. "
            "Add some documents and re-run."
        )

    print(f"Loaded {len(docs)} raw document sections.")
    return docs



def chunk_documents(docs):
    """
    Split documents into overlapping chunks.

    RecursiveCharacterTextSplitter tries to split on paragraph -> sentence
    -> word boundaries in that order, so chunks stay semantically coherent
    instead of cutting mid-sentence.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks "
          f"(size={config.CHUNK_SIZE}, overlap={config.CHUNK_OVERLAP}).")
    return chunks


def get_embeddings():
    if getattr(config, "USE_LOCAL_EMBEDDINGS", False) or not config.OPENAI_API_KEY:
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            return HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
        except ImportError:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            return HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
    return OpenAIEmbeddings(model=config.EMBEDDING_MODEL)


def build_vector_store(chunks):
    """Embed chunks and persist a FAISS index to disk."""
    embeddings = get_embeddings()
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(config.VECTOR_STORE_DIR)
    print(f"Vector store saved to '{config.VECTOR_STORE_DIR}/'.")
    return vector_store


if __name__ == "__main__":
    if not getattr(config, "USE_LOCAL_EMBEDDINGS", False) and not config.OPENAI_API_KEY:
        raise EnvironmentError(
            "Set OPENAI_API_KEY in a .env file before running ingestion, or set USE_LOCAL_EMBEDDINGS = True in config.py."
        )
    raw_docs = load_documents()
    doc_chunks = chunk_documents(raw_docs)
    build_vector_store(doc_chunks)

