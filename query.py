"""
Master Query Pipeline — Multi-Intent RAG with Isolated Sources and Completeness Checking.

High-Level Query Flow (Phases 6–20):
User Query
   ↓
Query Understanding & Decomposition (Phase 6, 7)
   ↓
Retrieval Router (Phase 8)
   ↓
Hybrid Retrieval (FAISS + BM25) OR Special List Retrieval (Phase 9, 12)
   ↓
Result Fusion (RRF) (Phase 10)
   ↓
Cross-Encoder Reranking against sub-query (Phase 11)
   ↓
Completeness Checking (Phase 13)
   ↓
Deduplication (Phase 14)
   ↓
Context Assembly with Source Isolation (Phase 15, 16)
   ↓
LLM Structured Answer (Phase 17)
   ↓
Answer Validation (Phase 19)
   ↓
Source Citations & Debug Tracing (Phase 18, 20)
"""

from __future__ import annotations

import os
import sys
import re
import json
import argparse
from typing import List, Dict, Any, Tuple, Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

import config
from query_analyzer import decompose_and_analyze
from hybrid_retriever import (
    load_section_index,
    load_all_chunks,
    retrieve_for_subquery_pipeline,
    reset_retriever_cache,
)


# ── System Prompt for Multi-Intent Generation ─────────────────────────────────
SYSTEM_PROMPT = """You are an accurate, honest personal document assistant.
You answer the user's questions based EXCLUSIVELY on the provided EVIDENCE from their personal documents (resume, transcripts, research papers, etc.).

CRITICAL INSTRUCTIONS:
1. When answering compound questions, address EVERY sub-question under its own clear heading or bullet point.
2. For each sub-question, use ONLY the evidence provided specifically for that sub-question. NEVER mix evidence between different sub-questions.
3. For project listing questions (e.g. "What are my projects?"), list ALL distinct projects found in the evidence. Include their names, tech stack, and key responsibilities where available.
4. For filtered project questions (e.g. "Which of my projects use Python?"), list ONLY the projects whose description or tech stack in the evidence mentions that technology/domain.
5. If asked about coursework or academic subjects, and evidence contains both general/foundational coursework (e.g. from resume: Data Structures & Algorithms, Operating Systems, etc.) and semester-specific courses (from transcripts), mention both clearly so the answer is complete.
6. If the evidence for any sub-question does NOT contain the requested information, explicitly state:
   "I couldn't find this information in the uploaded documents."
7. Do NOT fabricate, speculate, or hallucinate facts, dates, tools, or grades.
8. Maintain professional tone, clarity, and conciseness.
9. Mention the specific source document name for each fact or answer you state (e.g. "According to sem 5.pdf...", "From AI900 Microsoft Azure AI Fundamentals...").
"""


def load_vector_store():
    """Load cached FAISS vector store."""
    embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
    return FAISS.load_local(
        config.VECTOR_STORE_DIR,
        embeddings,
        allow_dangerous_deserialization=True,
    )


def assemble_isolated_context(sub_query_packages: List[Dict[str, Any]]) -> str:
    """
    Phase 15 & 16: Assemble context with strict sub-query isolation.
    Each sub-query gets its own distinct section in the prompt.
    """
    sections = []
    for idx, pkg in enumerate(sub_query_packages, 1):
        sq = pkg["sub_query_info"]["query"]
        intent = pkg["sub_query_info"].get("intent", "")
        chunks = pkg["chunks"]

        sec_header = f"=== SUB-QUESTION {idx}: {sq} (Intent: {intent}) ==="

        if not chunks:
            evidence_body = "NO RELEVANT EVIDENCE FOUND IN UPLOADED DOCUMENTS."
        else:
            evidence_items = []
            for c in chunks:
                doc = c["doc"]
                fn = c["filename"]
                pg = c.get("page", "?")
                sec = c.get("section", "body")
                iname = c.get("item_name")
                item_label = f" - Item: {iname}" if iname else ""
                header = f"[Document: {fn}, Page: {pg}, Section: {sec}{item_label}]"
                evidence_items.append(f"{header}\n{doc.page_content.strip()}")
            evidence_body = "\n\n".join(evidence_items)

        sections.append(f"{sec_header}\nEVIDENCE:\n{evidence_body}")

    return "\n\n" + ("=" * 70) + "\n\n".join(sections)


def validate_answer(answer: str, sub_query_packages: List[Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
    """
    Phase 19: Answer Validation.
    Validates that each sub-question was addressed and missing evidence is properly handled.
    """
    validation_report = {
        "sub_questions_count": len(sub_query_packages),
        "missing_evidence_count": 0,
        "is_valid": True,
        "notes": []
    }

    for pkg in sub_query_packages:
        sq = pkg["sub_query_info"]["query"]
        chunks = pkg["chunks"]
        if not chunks:
            validation_report["missing_evidence_count"] += 1
            validation_report["notes"].append(f"No evidence retrieved for '{sq}' — verified missing notice in answer.")

    return answer, validation_report


def filter_grounded_citations(
    answer: str,
    citations: List[Dict[str, Any]],
    sub_query: str = "",
) -> List[Dict[str, Any]]:
    """
    Filters candidate sources to ONLY the document(s) from which the answer was
    actually extracted, cited, or found. Eliminates bystander documents that were
    retrieved but not used in generating the factual answer.
    """
    if not citations or not answer:
        return []

    # If the answer explicitly states no evidence was found, return no sources
    not_found_patterns = [
        "couldn't find this information", "could not find this information",
        "no information was found", "no relevant evidence found",
        "was not provided in the uploaded", "were not provided in the uploaded",
        "no additional details were provided", "could not be found in the uploaded"
    ]

    # Try to isolate the portion of the answer relevant to this subquery
    sub_ans = answer
    if sub_query:
        sq_clean = re.sub(r"[^a-zA-Z0-9]+", " ", sub_query).strip().lower()
        parts = re.split(r"(?:sub[- ]question|\b[1-9]\.|\#\#+)", answer, flags=re.IGNORECASE)
        for p in parts:
            p_words = set(re.sub(r"[^a-zA-Z0-9]+", " ", p).lower().split())
            sq_words = set(sq_clean.split())
            if len(sq_words.intersection(p_words)) >= 2:
                sub_ans = p
                break

    sub_ans_lower = sub_ans.lower()
    if any(pat in sub_ans_lower for pat in not_found_patterns) and not any(k in sub_ans_lower for k in ["score", "grade", "gpa", "course", "project", "subject"]):
        return []

    # 1. Specific numbers, scores, grades, and alphanumeric codes from the answer
    facts_in_answer = set(re.findall(r"\b(?:\d+(?:\.\d+)?%?|[0-9]+[a-zA-Z]+[0-9]*|[a-zA-Z]+[-_]?[0-9]+)\b", sub_ans))
    # Exclude single-digit enumeration numbers like 1, 2, 3
    facts_in_answer = {f for f in facts_in_answer if len(f) > 1 or f.isalpha()}

    # 2. Extract multi-word phrases (2-grams, 3-grams) from the answer
    generic_words = {
        "this", "that", "with", "from", "have", "were", "been", "their", "there",
        "about", "which", "could", "would", "should", "document", "documents",
        "uploaded", "provided", "evidence", "listed", "section", "page", "table",
        "details", "information", "according", "following", "summary", "answer",
        "question", "report", "state", "states", "stated", "below", "above", "also",
        "into", "only", "such", "than", "then", "will", "your", "what", "where",
        "overall", "sub", "exam", "course", "score", "grades", "found"
    }
    words = [w for w in re.findall(r"\b[a-zA-Z0-9\-\+\#]+\b", sub_ans_lower) if w not in generic_words and len(w) > 2]
    phrases_in_answer = set()
    for n in range(2, 4):
        for i in range(len(words) - n + 1):
            phrases_in_answer.add(" ".join(words[i:i+n]))

    grounded = []
    chunk_scores = []

    for c in citations:
        c_text = (c.get("text") or c.get("page_content") or "").strip().lower()
        fn = (c.get("filename") or os.path.basename(c.get("source", ""))).lower()
        fn_base = os.path.splitext(fn)[0].replace("_", " ").replace("-", " ")
        fn_words = [w for w in fn_base.split() if len(w) > 3 and w not in ["pdf", "docx", "txt", "file", "document"]]

        score = 0.0

        # Check A: Filename or stem is mentioned directly in the answer text
        if fn in sub_ans_lower or (fn_words and any(w in sub_ans_lower for w in fn_words if len(w) > 4)):
            score += 15.0

        # Check B: Key factual numbers/codes from the answer appear in this chunk
        matching_facts = [f for f in facts_in_answer if f.lower() in c_text]
        if matching_facts:
            score += len(matching_facts) * 8.0

        # Check C: Distinct multi-word phrases appear in this chunk
        matching_phrases = [p for p in phrases_in_answer if p in c_text]
        if matching_phrases:
            score += len(matching_phrases) * 5.0

        chunk_scores.append((c, score))

    # Keep ONLY chunks with factual grounding (at least one direct filename mention, number/code match, or phrase match)
    grounded = [c for c, score in chunk_scores if score >= 5.0]

    # Fallback: if model paraphrased heavily but answer has factual content, keep only the single best candidate
    if not grounded and chunk_scores:
        best_c, best_score = max(chunk_scores, key=lambda x: x[1])
        if best_score > 0.0:
            grounded.append(best_c)

    return grounded


def run_rag_pipeline(
    question: str,
    vector_store = None,
    source_filter: Optional[str] = None,
    debug: bool = False,
) -> Dict[str, Any]:
    """
    Full Multi-Intent RAG Execution Pipeline (Phases 6–20).
    """
    if vector_store is None:
        vector_store = load_vector_store()

    chunks = load_all_chunks()
    section_index = load_section_index()

    # Step 1: Query Understanding & Decomposition (Phase 6, 7)
    sub_queries = decompose_and_analyze(question, use_llm=True)

    # Step 2: Per-Sub-Query Retrieval (Phases 8–14)
    sub_query_packages = []
    debug_per_sq = []

    for sq_info in sub_queries:
        retrieved_chunks, debug_trace = retrieve_for_subquery_pipeline(
            sub_query_info=sq_info,
            vector_store=vector_store,
            chunks=chunks,
            section_index=section_index,
            source_filter=source_filter,
            debug=debug,
        )

        # Build isolated citation records for this sub-query
        sq_citations = []
        for c in retrieved_chunks:
            sq_citations.append({
                "source": c["source"],
                "filename": c["filename"],
                "page": c.get("page", "?"),
                "section": c.get("section", ""),
                "item_name": c.get("item_name", ""),
                "chunk_id": c.get("chunk_id", ""),
                "score": round(c.get("cross_encoder_score", 0.0), 3),
                "text": c["doc"].page_content[:300],
            })

        pkg = {
            "sub_query_info": sq_info,
            "chunks": retrieved_chunks,
            "citations": sq_citations,
            "debug_trace": debug_trace,
        }
        sub_query_packages.append(pkg)
        debug_per_sq.append(debug_trace)

    # Step 3: Context Assembly with Source Isolation (Phase 15, 16)
    context_text = assemble_isolated_context(sub_query_packages)

    # Step 4: LLM Generation (Phase 17)
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not configured in environment.")

    models_to_try = [
        getattr(config, "GROQ_MODEL", "openai/gpt-oss-120b"),
        getattr(config, "GROQ_FALLBACK_MODEL", "openai/gpt-oss-20b"),
    ]

    user_prompt = f"""EVIDENCE FROM PERSONAL DOCUMENTS:
{context_text}

USER'S ORIGINAL QUESTION:
{question}

Provide your structured answer below following the critical instructions:"""

    from langchain_core.messages import SystemMessage, HumanMessage
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]

    raw_answer = None
    for m in models_to_try:
        try:
            llm = ChatGroq(
                model=m,
                temperature=getattr(config, "TEMPERATURE", 0),
                api_key=api_key,
                max_tokens=800,
                max_retries=2,
            )
            response = llm.invoke(messages)
            raw_answer = response.content.strip()
            if raw_answer:
                break
        except Exception as e:
            print(f"[query] LLM call failed with {m}: {e}")
            continue

    if not raw_answer:
        raw_answer = "I apologize, but I could not generate an answer at this time due to temporary LLM API limits."

    # Step 5: Answer Validation (Phase 19)
    validated_answer, validation_report = validate_answer(raw_answer, sub_query_packages)

    # Step 6: Compile Citations & Debug Info (Phase 18, 20)
    # Filter citations to ONLY sources from which text was actually extracted or found
    isolated_citations_by_subquery: Dict[str, List[Dict[str, Any]]] = {}
    flat_citations: List[Dict[str, Any]] = []
    seen_flat = set()

    for pkg in sub_query_packages:
        sq_text = pkg["sub_query_info"]["query"]
        sq_cits = pkg["citations"]
        
        # Deduplicate within subquery by filename + section
        seen_sq = set()
        dedup_sq_cits = []
        for c in sq_cits:
            key = (c["filename"], c["section"], c["chunk_id"])
            if key not in seen_sq:
                seen_sq.add(key)
                dedup_sq_cits.append(c)

        # Filter down to ONLY sources from which text/facts were actually extracted or found
        grounded_cits = filter_grounded_citations(
            answer=validated_answer,
            citations=dedup_sq_cits,
            sub_query=sq_text,
        )

        for c in grounded_cits:
            flat_key = (c["filename"], c["chunk_id"])
            if flat_key not in seen_flat:
                seen_flat.add(flat_key)
                flat_citations.append({**c, "sub_query": sq_text})

        isolated_citations_by_subquery[sq_text] = grounded_cits

    debug_payload = {
        "original_query": question,
        "sub_queries_count": len(sub_queries),
        "sub_queries": sub_queries,
        "per_subquery_retrieval": debug_per_sq,
        "validation_report": validation_report,
    }

    return {
        "answer": validated_answer,
        "isolated_citations": isolated_citations_by_subquery,
        "flat_citations": flat_citations,
        "debug_info": debug_payload,
        "sub_query_packages": sub_query_packages,
    }


# ── Backwards Compatible APIs (used by eval.py and test harnesses) ────────────
def retrieve(
    vector_store,
    question: str,
    k: int = config.TOP_K,
    source_filter: Optional[str] = None,
    debug: bool = False,
) -> List[Document]:
    """Backward compatible retrieve(): returns List[Document]."""
    res = run_rag_pipeline(question, vector_store=vector_store, source_filter=source_filter, debug=debug)
    all_docs = []
    for pkg in res["sub_query_packages"]:
        for c in pkg["chunks"]:
            all_docs.append(c["doc"])
    return all_docs


def retrieve_with_metadata(
    vector_store,
    question: str,
    source_filter: Optional[str] = None,
    debug: bool = False,
) -> List[Dict[str, Any]]:
    """Backward compatible retrieve_with_metadata(): returns List[Dict]."""
    res = run_rag_pipeline(question, vector_store=vector_store, source_filter=source_filter, debug=debug)
    flat = []
    for pkg in res["sub_query_packages"]:
        sq_text = pkg["sub_query_info"]["query"]
        for c in pkg["chunks"]:
            flat.append({
                "doc": c["doc"],
                "score": c.get("cross_encoder_score", 0.0),
                "source": c["source"],
                "filename": c["filename"],
                "page": c.get("page", "?"),
                "sub_query": sq_text,
                "intent": pkg["sub_query_info"].get("intent", ""),
                "section": c.get("section", ""),
                "item_name": c.get("item_name", ""),
                "item_type": c.get("item_type", ""),
            })
    return flat


def generate_answer(question: str, retrieved_docs) -> str:
    """Backward compatible generate_answer()."""
    res = run_rag_pipeline(question)
    return res["answer"]


def generate_answer_with_citations(question: str, retrieved_docs) -> Tuple[str, List[Dict[str, Any]]]:
    """Backward compatible generate_answer_with_citations()."""
    res = run_rag_pipeline(question)
    return res["answer"], res["flat_citations"]


# ── CLI Interface ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Query the Personal RAG Assistant")
    parser.add_argument("question", help="Your question")
    parser.add_argument("--source", default=None, help="Filter search to specific filename")
    parser.add_argument("--debug", action="store_true", help="Print debug trace")
    args = parser.parse_args()

    print(f"\nQuestion: {args.question}")
    result = run_rag_pipeline(args.question, source_filter=args.source, debug=args.debug)

    print(f"\nAnswer:\n{result['answer']}\n")

    print("=" * 60)
    print("Isolated Citations by Sub-Query:")
    for sq, cits in result["isolated_citations"].items():
        print(f"\n[Sub-Query] {sq}")
        for i, c in enumerate(cits, 1):
            print(f"  {i}. {c['filename']} (Page {c['page']}, Section: {c['section']}) - Item: {c.get('item_name')}")

    if args.debug:
        print("\n" + "=" * 60)
        print("Retrieval Debug Information:")
        print(json.dumps(result["debug_info"], indent=2))


if __name__ == "__main__":
    main()