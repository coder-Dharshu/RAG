"""
Phase 2, 3, 4, 5 — Document Ingestion Pipeline.

Pipeline:
Documents
   ↓
Document Parsing (via structure_parser.py)
   ↓
Structure Detection & Section Extraction
   ↓
Structure-Aware Chunking & Metadata Enrichment
   ↓
Dynamic Section Indexing (section_index.json)
   ↓
Embedding Generation
   ↓
FAISS Vector Index & Pickled Chunks for BM25

Run:
    python ingest.py
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import sys
import os
import glob
import json
import pickle

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
import config
from structure_parser import parse_document, build_section_index


def get_embeddings():
    """Load cached embedding model as configured."""
    if getattr(config, "USE_LOCAL_EMBEDDINGS", False) or not config.OPENAI_API_KEY:
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            return HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
        except ImportError:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            return HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings(model=config.EMBEDDING_MODEL)


from incremental_ingestion import rebuild_entire_index


def ingest_documents(docs_dir: str = config.DOCS_DIR):
    """
    Ingest all documents from docs_dir with structure-awareness, metadata enrichment,
    registry tracking, and section index generation.
    """
    print(f"\n{'='*60}")
    print(f"Starting Ingestion Pipeline from '{docs_dir}'")
    print(f"{'='*60}\n")

    result = rebuild_entire_index(docs_dir)

    from query import load_vector_store
    vector_store = load_vector_store()
    from hybrid_retriever import load_all_chunks, load_section_index
    all_chunks = load_all_chunks()
    section_index = load_section_index()

    print(f"\n{'='*60}")
    print(f"Ingestion Completed Successfully: {result['message']}")
    print(f"{'='*60}\n")
    return vector_store, all_chunks, section_index


if __name__ == "__main__":
    ingest_documents()
