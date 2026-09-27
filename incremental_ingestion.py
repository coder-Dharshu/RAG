"""
Phase 21, 22, 23 — Automatic Incremental Document Ingestion Pipeline.

Key Capabilities:
1. Stable SHA-256 file hashing for duplicate detection and document identity.
2. Ingestion Registry (document_registry.json) tracking hash, timestamp, chunk count, status.
3. Index Metadata (index_metadata.json) tracking model, dimensions, versions.
4. Three Ingestion Modes:
   - Case A (New file): Parse, chunk, enrich metadata, embed ONLY new chunks,
     append to FAISS via add_documents, update BM25 / chunks.pkl, update section index.
   - Case B (Duplicate - same name & hash): Skip or re-index by user request without duplication.
   - Case C (Modified - same name & new hash): Cleanly remove old chunks/vectors,
     ingest updated document without leaving stale chunks.
5. Deletion support: Safely remove a document and its chunks from retrieval.
6. Full index rebuild: Reprocesses all documents when requested or when configuration changes.
7. Atomic error handling: Transactional guarantees — failure to embed leaves document marked 'failed'
   rather than partially indexed.
"""

from __future__ import annotations

import os
import sys
import glob
import json
import pickle
import hashlib
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional, Callable, Union

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

import config
from structure_parser import parse_document, build_section_index

# ── Version constants ──────────────────────────────────────────────────────────
CHUNKING_VERSION = "structure-aware-v2"
INDEX_VERSION = "v3"

REGISTRY_FILENAME = "document_registry.json"
INDEX_METADATA_FILENAME = "index_metadata.json"
CHUNKS_FILENAME = "chunks.pkl"
SECTION_INDEX_FILENAME = "section_index.json"


# ── File Hashing ──────────────────────────────────────────────────────────────
def compute_file_hash(target: Union[str, bytes, Any]) -> str:
    """
    Computes a stable SHA-256 hash for a file path, raw bytes, or stream.
    """
    sha256 = hashlib.sha256()

    if isinstance(target, bytes):
        sha256.update(target)
    elif isinstance(target, str):
        if not os.path.exists(target):
            raise FileNotFoundError(f"File not found for hashing: '{target}'")
        with open(target, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                sha256.update(chunk)
    elif hasattr(target, "read"):
        # Stream-like object (e.g. Streamlit UploadedFile or BytesIO)
        pos = target.tell() if hasattr(target, "tell") else 0
        target.seek(0) if hasattr(target, "seek") else None
        for chunk in iter(lambda: target.read(65536), b""):
            sha256.update(chunk)
        if hasattr(target, "seek"):
            target.seek(pos)
    else:
        raise TypeError(f"Unsupported target type for hashing: {type(target)}")

    return sha256.hexdigest()


# ── Path Helpers ──────────────────────────────────────────────────────────────
def get_registry_path() -> str:
    return os.path.join(config.VECTOR_STORE_DIR, REGISTRY_FILENAME)


def get_index_metadata_path() -> str:
    return os.path.join(config.VECTOR_STORE_DIR, INDEX_METADATA_FILENAME)


def get_chunks_path() -> str:
    return os.path.join(config.VECTOR_STORE_DIR, CHUNKS_FILENAME)


def get_section_index_path() -> str:
    return os.path.join(config.VECTOR_STORE_DIR, SECTION_INDEX_FILENAME)


# ── Embedding Model Loader ────────────────────────────────────────────────────
def get_embeddings():
    """Load cached embedding model matching ingest.py and config."""
    if getattr(config, "USE_LOCAL_EMBEDDINGS", False) or not config.OPENAI_API_KEY:
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            return HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
        except ImportError:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            return HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings(model=config.EMBEDDING_MODEL)


# ── Persistence Helpers ───────────────────────────────────────────────────────
def load_document_registry() -> Dict[str, Any]:
    """
    Load document registry tracking indexed files.
    If missing, bootstraps registry from existing documents/ and chunks.pkl.
    """
    reg_path = get_registry_path()
    if os.path.exists(reg_path):
        try:
            with open(reg_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[incremental_ingestion] Warning loading registry: {e}")

    # Bootstrap registry if index already exists
    return bootstrap_registry_from_existing()


def save_document_registry(registry: Dict[str, Any]) -> None:
    """Save document registry atomically to disk."""
    os.makedirs(config.VECTOR_STORE_DIR, exist_ok=True)
    reg_path = get_registry_path()
    temp_path = reg_path + ".tmp"
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)
    if os.path.exists(reg_path):
        os.remove(reg_path)
    os.rename(temp_path, reg_path)


def load_index_metadata() -> Dict[str, Any]:
    """Load index configuration metadata (model, dimension, chunking version)."""
    meta_path = get_index_metadata_path()
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[incremental_ingestion] Warning loading index metadata: {e}")

    # Default metadata
    return {
        "embedding_model": config.EMBEDDING_MODEL,
        "embedding_dimension": 384 if "MiniLM" in config.EMBEDDING_MODEL else 1536,
        "chunking_version": CHUNKING_VERSION,
        "index_version": INDEX_VERSION,
        "created_at": datetime.now().isoformat(),
        "last_updated": datetime.now().isoformat(),
        "total_documents": 0,
        "total_chunks": 0,
    }


def save_index_metadata(meta: Dict[str, Any]) -> None:
    """Save index configuration metadata atomically to disk."""
    os.makedirs(config.VECTOR_STORE_DIR, exist_ok=True)
    meta_path = get_index_metadata_path()
    temp_path = meta_path + ".tmp"
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    if os.path.exists(meta_path):
        os.remove(meta_path)
    os.rename(temp_path, meta_path)


def bootstrap_registry_from_existing() -> Dict[str, Any]:
    """
    Populates document_registry.json and index_metadata.json from existing
    documents in documents/ and chunks.pkl so no re-embedding is required on startup.
    """
    registry: Dict[str, Any] = {}
    chunks_path = get_chunks_path()
    all_chunks: List[Document] = []

    if os.path.exists(chunks_path):
        try:
            with open(chunks_path, "rb") as f:
                all_chunks = pickle.load(f)
        except Exception as e:
            print(f"[incremental_ingestion] Error reading chunks.pkl: {e}")

    # Group existing chunks by filename
    chunk_counts: Dict[str, int] = {}
    doc_types: Dict[str, str] = {}
    doc_sections: Dict[str, List[str]] = {}

    for c in all_chunks:
        fn = c.metadata.get("filename") or os.path.basename(c.metadata.get("source", ""))
        if not fn:
            continue
        chunk_counts[fn] = chunk_counts.get(fn, 0) + 1
        if fn not in doc_types:
            doc_types[fn] = c.metadata.get("document_type", "unknown")
        if fn not in doc_sections:
            doc_sections[fn] = []
        sec = c.metadata.get("section")
        if sec and sec not in doc_sections[fn]:
            doc_sections[fn].append(sec)

    # Inspect documents directory
    if os.path.exists(config.DOCS_DIR):
        for entry in os.scandir(config.DOCS_DIR):
            if entry.is_file() and not entry.name.startswith("."):
                fn = entry.name
                fpath = entry.path
                try:
                    fhash = compute_file_hash(fpath)
                    fsize = entry.stat().st_size
                except Exception:
                    fhash = "unknown"
                    fsize = 0

                count = chunk_counts.get(fn, 0)
                status = "indexed" if count > 0 else "unindexed"

                registry[fn] = {
                    "hash": fhash,
                    "indexed_at": datetime.now().isoformat(),
                    "chunk_count": count,
                    "status": status,
                    "file_size": fsize,
                    "document_type": doc_types.get(fn, "unknown"),
                    "sections": doc_sections.get(fn, []),
                }

    save_document_registry(registry)

    # Save index metadata
    meta = {
        "embedding_model": config.EMBEDDING_MODEL,
        "embedding_dimension": 384 if "MiniLM" in config.EMBEDDING_MODEL else 1536,
        "chunking_version": CHUNKING_VERSION,
        "index_version": INDEX_VERSION,
        "created_at": datetime.now().isoformat(),
        "last_updated": datetime.now().isoformat(),
        "total_documents": len([r for r in registry.values() if r["status"] == "indexed"]),
        "total_chunks": len(all_chunks),
    }
    save_index_metadata(meta)
    print(f"[incremental_ingestion] Successfully bootstrapped registry with {len(registry)} documents.")
    return registry


# ── Configuration Compatibility Check ─────────────────────────────────────────
def check_index_compatibility() -> Tuple[bool, str]:
    """
    Verifies that the existing index on disk matches the active config
    (embedding model, chunking version, index version).
    Returns (is_compatible, message).
    """
    meta_path = get_index_metadata_path()
    if not os.path.exists(meta_path):
        return True, "No prior index metadata found (compatible)."

    meta = load_index_metadata()
    stored_model = meta.get("embedding_model")
    current_model = config.EMBEDDING_MODEL

    if stored_model and stored_model != current_model:
        return (
            False,
            f"Embedding model mismatch! Current config: '{current_model}', Index was built with: '{stored_model}'. "
            "A full index rebuild is required before uploading new documents."
        )

    stored_chunk_ver = meta.get("chunking_version")
    if stored_chunk_ver and stored_chunk_ver != CHUNKING_VERSION:
        return (
            False,
            f"Chunking version mismatch! Current: '{CHUNKING_VERSION}', Index: '{stored_chunk_ver}'. "
            "A full index rebuild is recommended."
        )

    return True, "Compatible"


# ── Duplicate & Modified Detection ────────────────────────────────────────────
def check_document_status(filename: str, file_bytes: bytes) -> Tuple[str, Dict[str, Any]]:
    """
    Evaluates upload status:
    - ("new", details): New file not present in registry.
    - ("duplicate", details): Exact same filename AND same hash already indexed.
    - ("modified", details): Same filename but different hash (document was updated).
    """
    file_hash = compute_file_hash(file_bytes)
    registry = load_document_registry()

    if filename in registry:
        reg_entry = registry[filename]
        existing_hash = reg_entry.get("hash", "")
        existing_status = reg_entry.get("status", "")

        if existing_hash == file_hash and existing_status == "indexed":
            return "duplicate", {
                "hash": file_hash,
                "registry_entry": reg_entry,
                "message": f"'{filename}' is already indexed (hash: {file_hash[:8]}...)."
            }
        else:
            return "modified", {
                "hash": file_hash,
                "old_hash": existing_hash,
                "registry_entry": reg_entry,
                "message": f"'{filename}' has been modified (new hash: {file_hash[:8]}...)."
            }

    # Also check if same hash exists under a different name
    for existing_name, entry in registry.items():
        if entry.get("hash") == file_hash and entry.get("status") == "indexed":
            return "duplicate", {
                "hash": file_hash,
                "registry_entry": entry,
                "message": f"Identical file content already indexed under '{existing_name}'."
            }

    return "new", {
        "hash": file_hash,
        "message": f"'{filename}' is a new document."
    }


# ── Incremental Document Ingestion ────────────────────────────────────────────
def ingest_new_document_incrementally(
    file_path: str,
    file_bytes: Optional[bytes] = None,
    vector_store: Optional[FAISS] = None,
    progress_callback: Optional[Callable[[str, float], None]] = None,
    reindex: bool = False,
) -> Dict[str, Any]:
    """
    Incrementally ingests a single document without re-embedding old documents.

    Pipeline Steps:
    1. Save file to DOCS_DIR & compute SHA-256 hash
    2. Check compatibility with active index configuration
    3. Detect duplicate vs modified vs new
    4. If modified or reindexing, remove previous document chunks and vectors
    5. Parse document into structure-aware chunks & attach full metadata
    6. Generate embeddings ONLY for new chunks & add to FAISS
    7. Update chunks.pkl & BM25 keyword retriever
    8. Update dynamic section index (section_index.json)
    9. Persist FAISS, registry, and metadata
    10. Document immediately becomes searchable
    """
    def notify(msg: str, progress: float):
        if progress_callback:
            progress_callback(msg, progress)
        print(f"[incremental_ingest] {msg}")

    filename = os.path.basename(file_path)
    os.makedirs(config.DOCS_DIR, exist_ok=True)
    os.makedirs(config.VECTOR_STORE_DIR, exist_ok=True)

    # Step 1: Save file if bytes provided
    notify(f"Processing '{filename}'...", 0.05)
    dest_path = os.path.join(config.DOCS_DIR, filename)

    if file_bytes is not None:
        with open(dest_path, "wb") as f:
            f.write(file_bytes)
        file_hash = compute_file_hash(file_bytes)
    else:
        if not os.path.exists(dest_path) and os.path.exists(file_path):
            import shutil
            shutil.copy2(file_path, dest_path)
        file_hash = compute_file_hash(dest_path)

    notify(f"✓ File saved to '{config.DOCS_DIR}' (SHA-256: {file_hash[:8]}...)", 0.15)

    # Step 2: Compatibility verification
    is_compat, compat_msg = check_index_compatibility()
    if not is_compat:
        raise ValueError(compat_msg)

    # Step 3: Status check
    registry = load_document_registry()
    status, status_info = check_document_status(filename, file_bytes or open(dest_path, "rb").read())

    if status == "duplicate" and not reindex:
        notify(f"Skipping: {filename} is already indexed.", 1.0)
        return {
            "status": "duplicate",
            "filename": filename,
            "hash": file_hash,
            "chunk_count": status_info["registry_entry"].get("chunk_count", 0),
            "message": f"'{filename}' is already indexed. No duplicate chunks created.",
        }

    # Step 4: Handle modified or re-indexed file (clean removal of old version)
    if status == "modified" or reindex:
        notify(f"Updating '{filename}': Removing previous version chunks...", 0.25)
        _remove_document_chunks_internal(filename, vector_store=vector_store)

    try:
        # Step 5: Document identification & structure parsing
        notify("✓ Document identified. Extracting text & detecting structure...", 0.35)
        new_chunks = parse_document(dest_path)

        if not new_chunks:
            raise ValueError(f"No text or chunks could be extracted from '{filename}'.")

        # Enrich chunk metadata with full traceability
        doc_type = new_chunks[0].metadata.get("document_type", "unknown")
        for chunk in new_chunks:
            chunk.metadata["file_hash"] = file_hash
            chunk.metadata["source"] = dest_path
            chunk.metadata["filename"] = filename
            chunk.metadata["text"] = chunk.page_content

        notify(f"✓ Structure detected: {len(new_chunks)} chunks created (Type: {doc_type})", 0.50)

        # Step 6: Load vector store and embeddings
        notify(f"✓ Generating {len(new_chunks)} embeddings (Sentence-Transformers)...", 0.65)
        embeddings = get_embeddings()

        if vector_store is None:
            from query import load_vector_store
            try:
                vector_store = load_vector_store()
            except Exception:
                vector_store = None

        if vector_store is None:
            # First time initialization
            vector_store = FAISS.from_documents(new_chunks, embeddings)
            notify("✓ FAISS vector store initialized with new vectors", 0.75)
        else:
            # INCREMENTAL ADD: Only embeds the new chunks!
            vector_store.add_documents(new_chunks)
            notify("✓ FAISS vector store incrementally updated with new vectors", 0.75)

        # Persist FAISS index immediately
        vector_store.save_local(config.VECTOR_STORE_DIR)
        notify("✓ FAISS vector index persisted to disk", 0.80)

        # Step 7: Update chunks.pkl
        chunks_path = get_chunks_path()
        all_chunks: List[Document] = []
        if os.path.exists(chunks_path):
            try:
                with open(chunks_path, "rb") as f:
                    all_chunks = pickle.load(f)
            except Exception:
                all_chunks = []

        # Filter out any old chunks for this file
        all_chunks = [c for c in all_chunks if c.metadata.get("filename") != filename]
        all_chunks.extend(new_chunks)

        with open(chunks_path, "wb") as f:
            pickle.dump(all_chunks, f)
        notify(f"✓ Updated chunk list persisted ({len(all_chunks)} total chunks)", 0.85)

        # Step 8: Update dynamic section index
        section_index = build_section_index(all_chunks)
        with open(get_section_index_path(), "w", encoding="utf-8") as f:
            json.dump(section_index, f, indent=2, ensure_ascii=False)
        notify("✓ Dynamic section index updated", 0.90)

        # Step 9: Reset retriever cache so BM25 & reranker immediately use updated corpus
        from hybrid_retriever import reset_retriever_cache
        reset_retriever_cache()
        notify("✓ BM25 keyword index and retriever caches re-initialized", 0.95)

        # Step 10: Update document registry & index metadata
        registry[filename] = {
            "hash": file_hash,
            "indexed_at": datetime.now().isoformat(),
            "chunk_count": len(new_chunks),
            "status": "indexed",
            "file_size": os.path.getsize(dest_path),
            "document_type": doc_type,
            "sections": list(set(c.metadata.get("section", "") for c in new_chunks if c.metadata.get("section"))),
        }
        save_document_registry(registry)

        meta = load_index_metadata()
        meta["total_chunks"] = len(all_chunks)
        meta["total_documents"] = len([r for r in registry.values() if r.get("status") == "indexed"])
        meta["last_updated"] = datetime.now().isoformat()
        if hasattr(vector_store, "index") and hasattr(vector_store.index, "d"):
            meta["embedding_dimension"] = vector_store.index.d
        save_index_metadata(meta)

        notify(f"Status: '{filename}' is now searchable!", 1.0)

        return {
            "status": "success",
            "filename": filename,
            "hash": file_hash,
            "chunk_count": len(new_chunks),
            "total_chunks": len(all_chunks),
            "document_type": doc_type,
            "message": f"'{filename}' was successfully indexed ({len(new_chunks)} chunks). It is now searchable.",
        }

    except Exception as e:
        # Atomic failure handling: mark document status as failed in registry
        import traceback
        err_detail = traceback.format_exc()
        print(f"[ERROR] Ingestion failed for '{filename}': {err_detail}")

        registry[filename] = {
            "hash": file_hash,
            "indexed_at": datetime.now().isoformat(),
            "chunk_count": 0,
            "status": "failed",
            "file_size": os.path.getsize(dest_path) if os.path.exists(dest_path) else 0,
            "error": str(e),
        }
        save_document_registry(registry)
        raise RuntimeError(f"Ingestion failed for '{filename}': {e}") from e


# ── Internal Chunk Removal Helper ─────────────────────────────────────────────
def _remove_document_chunks_internal(filename: str, vector_store: Optional[FAISS] = None) -> int:
    """
    Internal helper to remove all chunks and vectors belonging to filename.
    Does NOT delete the document from the registry (used when replacing/modifying).
    """
    chunks_path = get_chunks_path()
    removed_count = 0

    # 1. Update chunks.pkl
    if os.path.exists(chunks_path):
        with open(chunks_path, "rb") as f:
            all_chunks = pickle.load(f)

        remaining_chunks = [
            c for c in all_chunks
            if c.metadata.get("filename") != filename and os.path.basename(c.metadata.get("source", "")) != filename
        ]
        removed_count = len(all_chunks) - len(remaining_chunks)

        with open(chunks_path, "wb") as f:
            pickle.dump(remaining_chunks, f)

        # Update section index
        section_index = build_section_index(remaining_chunks)
        with open(get_section_index_path(), "w", encoding="utf-8") as f:
            json.dump(section_index, f, indent=2, ensure_ascii=False)

    # 2. Update FAISS vector store
    if vector_store is None:
        try:
            from query import load_vector_store
            vector_store = load_vector_store()
        except Exception:
            vector_store = None

    if vector_store is not None:
        try:
            doc_ids_to_delete = [
                k for k, v in vector_store.docstore._dict.items()
                if v.metadata.get("filename") == filename or os.path.basename(v.metadata.get("source", "")) == filename
            ]
            if doc_ids_to_delete:
                vector_store.delete(doc_ids_to_delete)
                vector_store.save_local(config.VECTOR_STORE_DIR)
                print(f"[incremental_ingest] Deleted {len(doc_ids_to_delete)} vectors from FAISS for '{filename}'.")
        except Exception as e:
            print(f"[incremental_ingest] Warning during vector deletion: {e}")
            # Fallback: rebuild FAISS from remaining chunks if delete fails
            if os.path.exists(chunks_path):
                with open(chunks_path, "rb") as f:
                    remaining_chunks = pickle.load(f)
                embeddings = get_embeddings()
                if remaining_chunks:
                    new_vs = FAISS.from_documents(remaining_chunks, embeddings)
                    new_vs.save_local(config.VECTOR_STORE_DIR)

    from hybrid_retriever import reset_retriever_cache
    reset_retriever_cache()
    return removed_count


# ── Delete Document from RAG ──────────────────────────────────────────────────
def remove_document_from_index(filename: str, vector_store: Optional[FAISS] = None) -> Dict[str, Any]:
    """
    Completely removes a document from the RAG index:
    - Removes vectors from FAISS
    - Removes chunks from chunks.pkl
    - Updates section index and BM25
    - Removes from document registry
    - Persists updated state
    """
    registry = load_document_registry()
    if filename not in registry:
        return {"status": "not_found", "message": f"'{filename}' is not in the document registry."}

    removed_count = _remove_document_chunks_internal(filename, vector_store=vector_store)

    # Remove from registry
    del registry[filename]
    save_document_registry(registry)

    # Update index metadata
    chunks_path = get_chunks_path()
    remaining_chunks_count = 0
    if os.path.exists(chunks_path):
        with open(chunks_path, "rb") as f:
            remaining_chunks_count = len(pickle.load(f))

    meta = load_index_metadata()
    meta["total_chunks"] = remaining_chunks_count
    meta["total_documents"] = len([r for r in registry.values() if r.get("status") == "indexed"])
    meta["last_updated"] = datetime.now().isoformat()
    save_index_metadata(meta)

    return {
        "status": "success",
        "filename": filename,
        "removed_chunks": removed_count,
        "remaining_chunks": remaining_chunks_count,
        "message": f"'{filename}' was successfully removed from RAG index.",
    }


# ── Full Index Rebuild ────────────────────────────────────────────────────────
def rebuild_entire_index(
    docs_dir: str = config.DOCS_DIR,
    progress_callback: Optional[Callable[[str, float], None]] = None,
) -> Dict[str, Any]:
    """
    Intentionally reprocesses ALL documents in docs_dir from scratch:
    - Re-chunks all documents with structure-aware parsing
    - Re-generates all embeddings
    - Recreates FAISS index and BM25
    - Rebuilds document registry with fresh hashes
    - Persists all state
    """
    def notify(msg: str, progress: float):
        if progress_callback:
            progress_callback(msg, progress)
        print(f"[full_rebuild] {msg}")

    notify(f"Starting full rebuild from '{docs_dir}'...", 0.05)

    supported_extensions = ("*.pdf", "*.txt", "*.md", "*.docx")
    files = []
    for ext in supported_extensions:
        files.extend(glob.glob(os.path.join(docs_dir, "**", ext), recursive=True))

    # Filter out hidden or temp files
    files = [f for f in files if not os.path.basename(f).startswith(".")]

    if not files:
        raise FileNotFoundError(f"No documents found in '{docs_dir}'.")

    all_chunks: List[Document] = []
    registry: Dict[str, Any] = {}
    total_files = len(files)

    for idx, fpath in enumerate(sorted(files), 1):
        fname = os.path.basename(fpath)
        fhash = compute_file_hash(fpath)
        fsize = os.path.getsize(fpath)
        step_progress = 0.10 + 0.40 * (idx / total_files)

        notify(f"Parsing '{fname}' ({idx}/{total_files})...", step_progress)
        try:
            chunks = parse_document(fpath)
            doc_type = chunks[0].metadata.get("document_type", "unknown") if chunks else "unknown"

            for c in chunks:
                c.metadata["file_hash"] = fhash
                c.metadata["source"] = fpath
                c.metadata["filename"] = fname
                c.metadata["text"] = c.page_content

            all_chunks.extend(chunks)
            registry[fname] = {
                "hash": fhash,
                "indexed_at": datetime.now().isoformat(),
                "chunk_count": len(chunks),
                "status": "indexed",
                "file_size": fsize,
                "document_type": doc_type,
                "sections": list(set(c.metadata.get("section", "") for c in chunks if c.metadata.get("section"))),
            }
        except Exception as e:
            print(f"[full_rebuild] Failed on '{fname}': {e}")
            registry[fname] = {
                "hash": fhash,
                "indexed_at": datetime.now().isoformat(),
                "chunk_count": 0,
                "status": "failed",
                "file_size": fsize,
                "error": str(e),
            }

    notify(f"Parsed {len(all_chunks)} chunks across {len(registry)} documents.", 0.55)

    # Dynamic section index
    notify("Building dynamic section index...", 0.65)
    section_index = build_section_index(all_chunks)
    os.makedirs(config.VECTOR_STORE_DIR, exist_ok=True)
    with open(get_section_index_path(), "w", encoding="utf-8") as f:
        json.dump(section_index, f, indent=2, ensure_ascii=False)

    # Persist chunks.pkl
    with open(get_chunks_path(), "wb") as f:
        pickle.dump(all_chunks, f)

    # Build FAISS vector store
    notify(f"Generating embeddings and building fresh FAISS vector store ({len(all_chunks)} chunks)...", 0.75)
    embeddings = get_embeddings()
    vector_store = FAISS.from_documents(all_chunks, embeddings)
    vector_store.save_local(config.VECTOR_STORE_DIR)

    # Save registry
    save_document_registry(registry)

    # Save index metadata
    dim = vector_store.index.d if hasattr(vector_store, "index") and hasattr(vector_store.index, "d") else 384
    meta = {
        "embedding_model": config.EMBEDDING_MODEL,
        "embedding_dimension": dim,
        "chunking_version": CHUNKING_VERSION,
        "index_version": INDEX_VERSION,
        "created_at": datetime.now().isoformat(),
        "last_updated": datetime.now().isoformat(),
        "total_documents": len([r for r in registry.values() if r.get("status") == "indexed"]),
        "total_chunks": len(all_chunks),
    }
    save_index_metadata(meta)

    # Reset retriever caches
    from hybrid_retriever import reset_retriever_cache
    reset_retriever_cache()

    notify("Full rebuild completed successfully!", 1.0)
    return {
        "status": "success",
        "total_documents": len(registry),
        "total_chunks": len(all_chunks),
        "vector_count": vector_store.index.ntotal,
        "message": f"Successfully rebuilt entire index ({len(all_chunks)} chunks across {len(registry)} files).",
    }
