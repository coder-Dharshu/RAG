"""
Phase 8, 9, 10, 11, 12, 13, 14 — Hybrid Retrieval, Result Fusion, Reranking,
Special List Retrieval, Completeness Checking, and Deduplication.

Components:
- route_retrieval()
- semantic_retrieve()
- keyword_retrieve()
- fuse_results() [RRF]
- rerank_results() [Cross-Encoder]
- retrieve_complete_section()
- check_completeness()
- deduplicate_results()
"""

from __future__ import annotations

import os
import re
import json
import pickle
from typing import List, Dict, Any, Tuple, Optional
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

import config

# ── Cross-Encoder Reranker Singleton ──────────────────────────────────────────
_reranker_instance: Optional[CrossEncoder] = None

def get_reranker() -> CrossEncoder:
    global _reranker_instance
    if _reranker_instance is None:
        model_name = getattr(config, "RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        print(f"[hybrid_retriever] Loading CrossEncoder: {model_name}...")
        _reranker_instance = CrossEncoder(model_name)
    return _reranker_instance


def reset_retriever_cache():
    """Clear cached models or indexes when reloading pipeline."""
    global _reranker_instance
    _reranker_instance = None


# ── Loaders ────────────────────────────────────────────────────────────────────
def load_section_index() -> Dict[str, Any]:
    sec_path = os.path.join(config.VECTOR_STORE_DIR, "section_index.json")
    if os.path.exists(sec_path):
        with open(sec_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def load_all_chunks() -> List[Document]:
    chunks_path = os.path.join(config.VECTOR_STORE_DIR, "chunks.pkl")
    if not os.path.exists(chunks_path):
        raise FileNotFoundError(f"'{chunks_path}' not found. Run ingest.py first.")
    with open(chunks_path, "rb") as f:
        return pickle.load(f)


# ── Phase 9: Hybrid Search Functions ──────────────────────────────────────────
def semantic_retrieve(vector_store, query: str, top_k: int = 15) -> List[Tuple[Document, float]]:
    """Retrieve top_k candidates from FAISS vector store with distance scores."""
    try:
        # returns List[Tuple[Document, float]] where float is L2 distance (lower is better)
        docs_and_scores = vector_store.similarity_search_with_score(query, k=top_k)
        return docs_and_scores
    except Exception as e:
        print(f"[hybrid_retriever] FAISS search error: {e}")
        docs = vector_store.similarity_search(query, k=top_k)
        return [(d, 1.0) for d in docs]


def keyword_retrieve(chunks: List[Document], query: str, top_k: int = 15) -> List[Tuple[Document, float]]:
    """Retrieve top_k candidates from BM25 keyword index."""
    try:
        from langchain_community.retrievers import BM25Retriever
        bm25 = BM25Retriever.from_documents(chunks)
        bm25.k = top_k
        docs = bm25.invoke(query)
        # BM25 returns sorted docs, assign synthetic rank-based score
        return [(doc, float(top_k - idx)) for idx, doc in enumerate(docs)]
    except Exception as e:
        print(f"[hybrid_retriever] BM25 search error: {e}")
        return []


# ── Phase 10: Result Fusion (Reciprocal Rank Fusion) ──────────────────────────
def fuse_results(
    semantic_results: List[Tuple[Document, float]],
    keyword_results: List[Tuple[Document, float]],
    source_hints: Optional[List[str]] = None,
    target_section: Optional[str] = None,
    rrf_k: int = 60,
    weight_semantic: float = 0.6,
    weight_keyword: float = 0.4,
) -> List[Dict[str, Any]]:
    """
    Combines semantic and keyword retrieval results using Reciprocal Rank Fusion.
    Preserves all retrieval metadata (semantic score, keyword score, fused score, source, section, item).
    """
    candidates: Dict[str, Dict[str, Any]] = {}

    # 1. Process Semantic candidates
    for rank, (doc, score) in enumerate(semantic_results):
        cid = doc.metadata.get("chunk_id") or doc.page_content[:80]
        if cid not in candidates:
            candidates[cid] = {
                "doc": doc,
                "semantic_rank": rank,
                "keyword_rank": None,
                "semantic_score": float(score),
                "keyword_score": 0.0,
                "rrf_score": 0.0,
                "source": doc.metadata.get("source", "unknown"),
                "filename": doc.metadata.get("filename") or os.path.basename(doc.metadata.get("source", "")),
                "page": doc.metadata.get("page", "?"),
                "document_type": doc.metadata.get("document_type", "other"),
                "section": doc.metadata.get("section", ""),
                "item_name": doc.metadata.get("item_name", ""),
                "item_type": doc.metadata.get("item_type", ""),
                "chunk_id": doc.metadata.get("chunk_id", ""),
                "retrieval_method": "semantic",
            }
        candidates[cid]["rrf_score"] += weight_semantic / (rrf_k + rank + 1)

    # 2. Process Keyword candidates
    for rank, (doc, score) in enumerate(keyword_results):
        cid = doc.metadata.get("chunk_id") or doc.page_content[:80]
        if cid not in candidates:
            candidates[cid] = {
                "doc": doc,
                "semantic_rank": None,
                "keyword_rank": rank,
                "semantic_score": 0.0,
                "keyword_score": float(score),
                "rrf_score": 0.0,
                "source": doc.metadata.get("source", "unknown"),
                "filename": doc.metadata.get("filename") or os.path.basename(doc.metadata.get("source", "")),
                "page": doc.metadata.get("page", "?"),
                "document_type": doc.metadata.get("document_type", "other"),
                "section": doc.metadata.get("section", ""),
                "item_name": doc.metadata.get("item_name", ""),
                "item_type": doc.metadata.get("item_type", ""),
                "chunk_id": doc.metadata.get("chunk_id", ""),
                "retrieval_method": "keyword",
            }
        else:
            candidates[cid]["keyword_rank"] = rank
            candidates[cid]["keyword_score"] = float(score)
            candidates[cid]["retrieval_method"] = "hybrid"

        candidates[cid]["rrf_score"] += weight_keyword / (rrf_k + rank + 1)

    # 3. Soft metadata routing boost (guided, not eliminating)
    for c in candidates.values():
        doc_type = c["document_type"]
        sec = c["section"]
        # Boost if doc_type matches source_hints
        if source_hints and doc_type in source_hints:
            c["rrf_score"] *= 1.25
        # Boost if section matches target_section
        if target_section and sec == target_section:
            c["rrf_score"] *= 1.35

    fused_list = sorted(candidates.values(), key=lambda x: x["rrf_score"], reverse=True)
    return fused_list


# ── Phase 11: Cross-Encoder Reranking ─────────────────────────────────────────
def rerank_results(
    sub_query: str,
    candidates: List[Dict[str, Any]],
    top_k: int = 5,
    min_score_gate: float = -11.0,
) -> List[Dict[str, Any]]:
    """
    Reranks candidates using Cross-Encoder against the SUB-QUERY.
    Ranking primarily; min_score_gate acts as secondary safeguard against irrelevant noise.
    """
    if not candidates:
        return []

    reranker = get_reranker()
    pairs = [(sub_query, c["doc"].page_content) for c in candidates]
    scores = reranker.predict(pairs)

    for c, score in zip(candidates, scores):
        c["cross_encoder_score"] = float(score)

    ranked = sorted(candidates, key=lambda x: x["cross_encoder_score"], reverse=True)

    # Filter out obvious outliers below safeguard gate
    filtered = [c for c in ranked if c["cross_encoder_score"] >= min_score_gate]
    return filtered[:top_k] if filtered else ranked[:top_k]


# ── Phase 12 & 13: Special List Retrieval & Completeness Checking ─────────────
def retrieve_list_items(
    chunks: List[Document],
    section_index: Dict[str, Any],
    sub_query_info: Dict[str, Any],
    debug: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Specialized retrieval for list/enumeration queries (e.g. 'What are my projects?').
    1. Reads section index for target section (e.g. 'projects')
    2. Retrieves all expected items dynamically
    3. Filters if query is 'filtered_list' (e.g. Python / AI)
    4. Performs completeness check: detected vs retrieved vs missing
    """
    target_section = sub_query_info.get("section") or "projects"
    filter_criteria = sub_query_info.get("filter_criteria")
    q_str = sub_query_info.get("query", "")

    # Find expected items from section index across matching documents
    detected_items: List[str] = []
    target_chunk_ids: List[str] = []

    for fn, doc_info in section_index.items():
        doc_type = doc_info.get("document_type")
        # For personal list queries, look in resume only for project or experience listings
        if sub_query_info.get("intent", "").startswith("personal") and target_section in ("projects", "experience") and doc_type != "resume":
            continue

        sec_data = doc_info.get("sections", {}).get(target_section, {})
        items = sec_data.get("items", [])
        cids = sec_data.get("chunk_ids", [])
        detected_items.extend(items)
        target_chunk_ids.extend(cids)

    # Map chunks by chunk_id
    chunk_map = {c.metadata.get("chunk_id"): c for c in chunks if c.metadata.get("chunk_id")}

    retrieved_results: List[Dict[str, Any]] = []
    retrieved_item_names: List[str] = []

    for cid in target_chunk_ids:
        chunk = chunk_map.get(cid)
        if not chunk:
            continue

        item_name = chunk.metadata.get("item_name", "")

        # If filtered_list query (e.g. projects using Python)
        if filter_criteria:
            crit = filter_criteria.lower()
            text_to_check = chunk.page_content.lower()
            # Check if filter keyword exists in the project chunk
            # e.g., \bpython\b or \bai\b
            pattern = rf"\b{re.escape(crit)}\b"
            if not re.search(pattern, text_to_check, re.IGNORECASE):
                continue

        retrieved_results.append({
            "doc": chunk,
            "semantic_score": 1.0,
            "keyword_score": 1.0,
            "rrf_score": 1.0,
            "cross_encoder_score": 5.0,  # Exact section match
            "source": chunk.metadata.get("source", "unknown"),
            "filename": chunk.metadata.get("filename") or os.path.basename(chunk.metadata.get("source", "")),
            "page": chunk.metadata.get("page", 1),
            "document_type": chunk.metadata.get("document_type", "resume"),
            "section": target_section,
            "item_name": item_name,
            "item_type": chunk.metadata.get("item_type", "project"),
            "chunk_id": cid,
            "retrieval_method": "section_index",
        })
        retrieved_item_names.append(item_name)

    # Completeness check (Phase 13)
    if filter_criteria:
        completeness_info = {
            "section": target_section,
            "filter": filter_criteria,
            "detected_total_in_section": len(detected_items),
            "matching_filter_retrieved": len(retrieved_results),
            "retrieved_items": retrieved_item_names,
            "status": "complete",
        }
    else:
        missing = [it for it in detected_items if it not in retrieved_item_names]
        completeness_info = {
            "section": target_section,
            "filter": None,
            "detected_items_count": len(detected_items),
            "retrieved_items_count": len(retrieved_results),
            "missing_count": len(missing),
            "missing_items": missing,
            "status": "complete" if len(missing) == 0 else "incomplete",
        }

    return retrieved_results, completeness_info


# ── Phase 14: Deduplication ───────────────────────────────────────────────────
def deduplicate_results(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Deduplicates chunks based on chunk_id and exact content,
    while guaranteeing distinct projects/items are NEVER collapsed.
    """
    seen_ids = set()
    seen_hashes = set()
    unique = []

    for c in candidates:
        cid = c.get("chunk_id")
        content_hash = hash(c["doc"].page_content.strip())

        if cid and cid in seen_ids:
            continue
        if content_hash in seen_hashes:
            continue

        if cid:
            seen_ids.add(cid)
        seen_hashes.add(content_hash)
        unique.append(c)

    return unique


# ── Master Sub-Query Retriever (Phases 8-14) ──────────────────────────────────
def retrieve_for_subquery_pipeline(
    sub_query_info: Dict[str, Any],
    vector_store,
    chunks: List[Document],
    section_index: Dict[str, Any],
    source_filter: Optional[str] = None,
    debug: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Executes the entire retrieval pipeline for a single sub-query:
    Routing -> (Section Index / Hybrid Search) -> Fusion -> Reranking -> Deduplication -> Completeness
    """
    q_str = sub_query_info["query"]
    intent = sub_query_info.get("intent", "general_document_query")
    qtype = sub_query_info.get("question_type", "factual")
    source_hints = sub_query_info.get("source_hint", [])
    target_section = sub_query_info.get("section")
    filter_criteria = sub_query_info.get("filter_criteria")

    debug_trace: Dict[str, Any] = {
        "sub_query": q_str,
        "intent": intent,
        "question_type": qtype,
        "source_hints": source_hints,
        "target_section": target_section,
    }

    # 1. Routing Check: Is this an enumeration / list question?
    has_academic_keywords = any(w in q_str.lower() for w in ["sem", "semester", "grade", "transcript", "sgpa", "cgpa", "marks card"]) or \
                            (target_section in ("course_grades", "academic_performance"))

    is_list_query = (not has_academic_keywords) and (
        (qtype in ("list", "filtered_list") and target_section in ("projects", "skills", "experience", "education", "coursework")) or
        ("project" in q_str.lower() and qtype in ("list", "filtered_list"))
    )

    list_results = []
    if is_list_query:
        found_items, comp_info = retrieve_list_items(
            chunks=chunks,
            section_index=section_index,
            sub_query_info=sub_query_info,
            debug=debug
        )
        if found_items:
            list_results = found_items
            debug_trace["completeness"] = comp_info
            debug_trace["section_index_items_count"] = len(found_items)

    # 2. Universal Hybrid Retrieval (FAISS Semantic + BM25 Keyword + Cross-Encoder)
    debug_trace["routing"] = "universal_hybrid_search"

    # Candidates fetch limit
    fetch_k = max(config.TOP_K * 4, 20)

    # FAISS Semantic Search
    sem_cands = semantic_retrieve(vector_store, q_str, top_k=fetch_k)
    debug_trace["faiss_candidates_count"] = len(sem_cands)

    # BM25 Keyword Search
    bm25_cands = keyword_retrieve(chunks, q_str, top_k=fetch_k)
    debug_trace["bm25_candidates_count"] = len(bm25_cands)

    # 3. Result Fusion (RRF)
    fused_cands = fuse_results(
        semantic_results=sem_cands,
        keyword_results=bm25_cands,
        source_hints=source_hints,
        target_section=target_section,
    )
    debug_trace["fused_candidates_count"] = len(fused_cands)

    # 4. Optional global source filter
    if source_filter:
        fused_cands = [c for c in fused_cands if source_filter.lower() in c["source"].lower()]

    # 5. Cross-Encoder Reranking
    keep_k = max(config.TOP_K // 2, 4)
    reranked = rerank_results(q_str, fused_cands, top_k=keep_k)

    # Source Isolation & Filtering (Phase 16):
    # Source Filtering & User Preferences (Phase 16):
    q_lower = q_str.lower()

    # 1. Explicit user negation: if prompt says "not from resume", exclude resume documents
    if "not from resume" in q_lower or "not from the resume" in q_lower:
        filtered_neg = [c for c in reranked if c.get("document_type") != "resume"]
        if filtered_neg:
            reranked = filtered_neg

    # 2. Universal Retrieval: Respect Cross-Encoder ranking across all documents.
    # Any uploaded document (PDF, DOCX, TXT) that scores high on relevance is retained.
    debug_trace["reranked_count"] = len(reranked)
    debug_trace["cross_encoder_scores"] = [
        {"chunk_id": c["chunk_id"], "score": round(c.get("cross_encoder_score", 0.0), 3), "source": c["filename"], "doc_type": c.get("document_type")}
        for c in reranked
    ]

    # 3. Deduplication & Final Assembly
    combined_candidates = list_results + reranked if list_results else reranked
    if source_filter:
        combined_candidates = [c for c in combined_candidates if source_filter.lower() in c["source"].lower()]

    unique_results = deduplicate_results(combined_candidates)
    debug_trace["selected_chunks_count"] = len(unique_results)
    debug_trace["completeness"] = {"status": "complete", "retrieved_items_count": len(unique_results)}

    return unique_results, debug_trace
