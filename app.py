"""
Streamlit UI for the Multi-Intent Production Personal RAG Assistant.

Architecture & Features:
- Natural chat interface with persistent message history
- Automatic Incremental Document Ingestion (Upload -> Hash -> Parse -> Embed -> FAISS/BM25)
- Duplicate detection (Skip / Re-index) & Modified document replacement
- Document management panel (Indexed documents list with Re-index & Delete)
- Collapsible "🔍 Index Status" system diagnostics panel
- Explicit "🔄 Rebuild Entire Index" vs "🔄 Reload pipeline"
- Per-sub-query source isolation and completeness checking
- Collapsible "🔍 Retrieval Debug" inspection panel
- Dynamic source filter dropdown
"""

from __future__ import annotations

import os
import sys
import json
import math
import importlib
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

import config
from incremental_ingestion import (
    compute_file_hash,
    load_document_registry,
    load_index_metadata,
    check_index_compatibility,
    check_document_status,
    ingest_new_document_incrementally,
    remove_document_from_index,
    rebuild_entire_index,
    INDEX_VERSION,
    CHUNKING_VERSION,
)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Personal RAG Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stMetric"] {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border: 1px solid #0f3460; border-radius: 12px; padding: 14px;
}
[data-testid="stMetricValue"] { font-size: 1.8rem !important; font-weight: 700; }
[data-testid="stFileUploader"] {
    border: 2px dashed rgba(99,102,241,0.4);
    border-radius: 12px; padding: 8px;
}
.stTabs [data-baseweb="tab"] { font-size: 1rem; font-weight: 600; padding: 10px 20px; }
.stTabs [aria-selected="true"] { border-bottom: 3px solid #6366f1; }
</style>
""", unsafe_allow_html=True)

# ── Pipeline Loader (Cached) ───────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading pipeline and vector store...")
def load_pipeline():
    from query import load_vector_store
    return load_vector_store()


@st.cache_data(show_spinner=False)
def get_available_sources():
    from hybrid_retriever import load_section_index
    section_index = load_section_index()
    if section_index:
        return sorted(list(section_index.keys()))

    chunks_path = os.path.join(config.VECTOR_STORE_DIR, "chunks.pkl")
    if not os.path.exists(chunks_path):
        return []
    import pickle
    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)
    return sorted(set(
        os.path.basename(c.metadata.get("source", ""))
        for c in chunks if c.metadata.get("source")
    ))


# Initialize or retrieve vector store in session state
try:
    if "vector_store" not in st.session_state or st.session_state.vector_store is None:
        st.session_state.vector_store = load_pipeline()
    vector_store = st.session_state.vector_store
except Exception as e:
    st.error(f"Could not load vector store: {e}")
    st.info("Run `python ingest.py` or use 'Rebuild Entire Index' below.")
    vector_store = None


def reload_system_pipeline():
    """Reloads modules, retriever caches, and vector store without reprocessing documents."""
    for mod_name in [
        "config", "document_classifier", "structure_parser", "incremental_ingestion",
        "query_analyzer", "hybrid_retriever", "query"
    ]:
        if mod_name in sys.modules:
            importlib.reload(sys.modules[mod_name])

    from hybrid_retriever import reset_retriever_cache
    reset_retriever_cache()

    st.cache_resource.clear()
    st.cache_data.clear()
    try:
        from query import load_vector_store
        st.session_state.vector_store = load_vector_store()
    except Exception:
        pass


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📚 RAG Assistant")
    st.caption("Personal Document Intelligence System")

    # Incompatibility Warning Check
    is_compat, compat_msg = check_index_compatibility()
    if not is_compat:
        st.warning(f"⚠️ **Index Incompatibility**: {compat_msg}")

    # ── 1. Document Upload & Ingestion Section ──────────────────────────────────
    st.markdown("### 📤 Upload Document")
    st.caption("Auto-indexes into FAISS & BM25 incrementally without re-embedding old documents.")

    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["pdf", "docx", "txt", "md"],
        key="doc_uploader",
        help="Upload PDF, DOCX, TXT, or MD documents. Automatically chunked, embedded, and indexed."
    )

    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        fname = uploaded_file.name

        status, status_info = check_document_status(fname, file_bytes)

        if status == "duplicate":
            st.info(f"📄 **{fname}** is already indexed ({status_info['registry_entry'].get('chunk_count', 0)} chunks).")
            col_skip, col_reidx = st.columns(2)
            with col_skip:
                if st.button("⏭️ Skip", key=f"btn_skip_{fname}", use_container_width=True):
                    st.toast(f"Skipped {fname}")
            with col_reidx:
                if st.button("🔄 Re-index", key=f"btn_reidx_{fname}", type="primary", use_container_width=True):
                    with st.status(f"Re-indexing '{fname}'...", expanded=True) as status_box:
                        p_bar = st.progress(0.0)
                        def on_prog(msg, p):
                            st.write(msg)
                            p_bar.progress(p)

                        dest = os.path.join(config.DOCS_DIR, fname)
                        res = ingest_new_document_incrementally(
                            file_path=dest,
                            file_bytes=file_bytes,
                            vector_store=vector_store,
                            progress_callback=on_prog,
                            reindex=True,
                        )
                        status_box.update(label=f"✓ '{fname}' re-indexed successfully!", state="complete", expanded=False)
                        st.success(f"🎉 **{fname}** re-indexed ({res['chunk_count']} chunks).")
                        reload_system_pipeline()
                        st.rerun()

        else:
            # New or Modified file: Automatic Ingestion Flow
            action_label = "Updating modified" if status == "modified" else "Ingesting new"
            with st.status(f"{action_label} document '{fname}'...", expanded=True) as status_box:
                p_bar = st.progress(0.0)
                def on_prog(msg, p):
                    st.write(msg)
                    p_bar.progress(p)

                dest = os.path.join(config.DOCS_DIR, fname)
                try:
                    res = ingest_new_document_incrementally(
                        file_path=dest,
                        file_bytes=file_bytes,
                        vector_store=vector_store,
                        progress_callback=on_prog,
                        reindex=(status == "modified"),
                    )
                    status_box.update(label=f"✓ '{fname}' is now searchable!", state="complete", expanded=False)
                    st.success(f"🎉 **{fname}** is now searchable! ({res['chunk_count']} chunks indexed)")
                    reload_system_pipeline()
                    st.rerun()
                except Exception as e:
                    status_box.update(label=f"❌ Failed to ingest '{fname}'", state="error", expanded=True)
                    st.error(f"Error: {e}")

    st.markdown("---")

    # ── 2. Source Filter ───────────────────────────────────────────────────────
    st.markdown("### 🔍 Search Scope")
    sources = get_available_sources()
    source_options = ["All documents"] + sources
    selected_source = st.selectbox("Search in:", options=source_options, index=0)
    source_filter = None if selected_source == "All documents" else selected_source

    st.markdown("---")

    # ── 3. Document Management Panel ───────────────────────────────────────────
    st.markdown("### 📁 Indexed Documents")
    registry = load_document_registry()
    indexed_docs = {k: v for k, v in registry.items() if v.get("status") == "indexed"}

    if not indexed_docs:
        st.caption("No documents indexed yet.")
    else:
        for doc_name, doc_info in list(indexed_docs.items()):
            cnt = doc_info.get("chunk_count", 0)
            dtype = doc_info.get("document_type", "document").replace("_", " ").title()
            with st.container():
                st.markdown(f"**{doc_name}**")
                c_info, c_acts = st.columns([3, 2])
                with c_info:
                    st.caption(f"{cnt} chunks · {dtype}")
                with c_acts:
                    col_re, col_del = st.columns(2)
                    with col_re:
                        if st.button("🔄", key=f"reidx_item_{doc_name}", help=f"Re-index {doc_name}"):
                            doc_path = os.path.join(config.DOCS_DIR, doc_name)
                            if os.path.exists(doc_path):
                                with st.spinner(f"Re-indexing {doc_name}..."):
                                    ingest_new_document_incrementally(
                                        file_path=doc_path,
                                        vector_store=vector_store,
                                        reindex=True,
                                    )
                                    reload_system_pipeline()
                                    st.toast(f"Re-indexed {doc_name}")
                                    st.rerun()
                    with col_del:
                        if st.button("🗑️", key=f"del_item_{doc_name}", help=f"Remove {doc_name} from RAG"):
                            with st.spinner(f"Removing {doc_name}..."):
                                remove_document_from_index(doc_name, vector_store=vector_store)
                                reload_system_pipeline()
                                st.toast(f"Removed {doc_name}")
                                st.rerun()

    st.markdown("---")

    # ── 4. Diagnostics & Debug Options ─────────────────────────────────────────
    st.markdown("### ⚙️ System Controls")

    if st.button("🗑️ Clear chat history", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    if st.button(
        "🔄 Reload pipeline",
        use_container_width=True,
        help="Reloads configuration, document metadata, section index, FAISS vector index, and BM25 models without re-embedding."
    ):
        reload_system_pipeline()
        st.success("Pipeline, indexes, and models reloaded successfully!")
        st.rerun()

    debug_mode = st.toggle(
        "Show retrieval debug info",
        value=False,
        help="Expands step-by-step decomposition, candidate scoring, and completeness checking."
    )

    # ── 5. Rebuild Entire Index ────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### ⚠️ Rebuild Entire Index")
    st.caption("Reprocesses ALL documents from scratch. Use only when changing embedding model or chunking logic.")

    if "confirm_rebuild" not in st.session_state:
        st.session_state.confirm_rebuild = False

    if not st.session_state.confirm_rebuild:
        if st.button("🔄 Rebuild Entire Index", use_container_width=True):
            st.session_state.confirm_rebuild = True
            st.rerun()
    else:
        st.warning("⚠️ This will reprocess all documents and regenerate the FAISS/BM25 indexes.")
        c_yes, c_no = st.columns(2)
        with c_yes:
            if st.button("Yes, Rebuild", type="primary", use_container_width=True):
                with st.status("Rebuilding entire index from scratch...", expanded=True) as s_box:
                    p_bar = st.progress(0.0)
                    def rebuild_prog(msg, p):
                        st.write(msg)
                        p_bar.progress(p)

                    res = rebuild_entire_index(progress_callback=rebuild_prog)
                    s_box.update(label="✓ Entire index rebuilt!", state="complete", expanded=False)
                    st.success(res["message"])
                    st.session_state.confirm_rebuild = False
                    reload_system_pipeline()
                    st.rerun()
        with c_no:
            if st.button("Cancel", use_container_width=True):
                st.session_state.confirm_rebuild = False
                st.rerun()

    # ── 6. Index Status Expander (Diagnostics) ─────────────────────────────────
    st.markdown("---")
    with st.expander("🔍 Index Status"):
        meta = load_index_metadata()
        total_docs = len(indexed_docs)
        total_chunks = meta.get("total_chunks", 0)
        faiss_vecs = (vector_store.index.ntotal
                      if (vector_store and hasattr(vector_store, "index"))
                      else total_chunks)
        st.markdown(f"**Total documents:** {total_docs}")
        st.markdown(f"**Total chunks:** {total_chunks}")
        st.markdown(f"**FAISS vectors:** {faiss_vecs}")
        st.markdown(f"**Embedding model:** `{meta.get('embedding_model')}`")
        st.markdown(f"**Embedding dim:** {meta.get('embedding_dimension', 384)}")
        st.markdown(f"**Index version:** `{meta.get('index_version', INDEX_VERSION)}`")
        st.markdown(f"**Last indexed:** {meta.get('last_updated', 'N/A')[:19]}")
        st.markdown("---")
        st.markdown("**Registry:**")
        for dname, dinfo in registry.items():
            st.markdown(f"• **{dname}**")
            st.caption(
                f"Hash: `{dinfo.get('hash', '')[:12]}...` | "
                f"Chunks: {dinfo.get('chunk_count', 0)} | "
                f"Status: `{dinfo.get('status')}`"
            )

st.title("📚 Personal RAG Assistant")

tab_chat, tab_upload, tab_eval = st.tabs([
    "💬  Chat",
    "📤  Upload Documents",
    "📊  RAGAS Evaluation",
])


# =============================================================================
# TAB 1 — CHAT
# =============================================================================
with tab_chat:
    st.markdown(
        "Ask questions about your documents — resume, academic transcripts, "
        "research papers, and newly uploaded files."
    )

    if source_filter:
        st.info(f"🔍 Searching strictly in: **{source_filter}**", icon="📄")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                isolated_cits = message.get("isolated_citations", {})
                debug_info = message.get("debug_info")

                total_cits = sum(len(cits) for cits in isolated_cits.values()) if isolated_cits else 0
                if total_cits > 0:
                    with st.expander(f"📎 Sources ({total_cits} reference{'s' if total_cits > 1 else ''})"):
                        for sq, cits in isolated_cits.items():
                            if not cits:
                                continue
                            if len(isolated_cits) > 1:
                                st.markdown(f"**🔍 {sq}**")
                            for c in cits:
                                fn  = c.get("filename", os.path.basename(c.get("source", "?")))
                                pg  = c.get("page", "?")
                                sec = c.get("section", "")
                                sec_label  = f" — *{sec}*" if sec else ""
                                item_label = f" (*{c['item_name']}*)" if c.get("item_name") else ""
                                st.markdown(f"• **{fn}** — page {pg}{sec_label}{item_label}")
                                st.caption(f'"{c["text"].strip()[:240]}…"')
                            st.divider()

                if debug_info and debug_mode:
                    with st.expander("🔍 Retrieval Debug"):
                        st.json(debug_info)

    if prompt := st.chat_input("Ask something about your documents…"):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            with st.spinner("Analyzing query, retrieving evidence & generating answer…"):
                try:
                    from query import run_rag_pipeline
                    result = run_rag_pipeline(
                        question=prompt,
                        vector_store=vector_store,
                        source_filter=source_filter,
                        debug=debug_mode,
                    )
                    answer        = result["answer"]
                    isolated_cits = result["isolated_citations"]
                    debug_info    = result["debug_info"]
                except Exception as e:
                    import traceback
                    answer        = f"An error occurred: {e}"
                    isolated_cits = {}
                    debug_info    = {"error": traceback.format_exc()}

            st.markdown(answer)

            total_cits = sum(len(cits) for cits in isolated_cits.values()) if isolated_cits else 0
            if total_cits > 0:
                with st.expander(f"📎 Sources ({total_cits} reference{'s' if total_cits > 1 else ''})"):
                    for sq, cits in isolated_cits.items():
                        if not cits:
                            continue
                        if len(isolated_cits) > 1:
                            st.markdown(f"**🔍 {sq}**")
                        for c in cits:
                            fn  = c.get("filename", os.path.basename(c.get("source", "?")))
                            pg  = c.get("page", "?")
                            sec = c.get("section", "")
                            sec_label  = f" — *{sec}*" if sec else ""
                            item_label = f" (*{c['item_name']}*)" if c.get("item_name") else ""
                            st.markdown(f"• **{fn}** — page {pg}{sec_label}{item_label}")
                            st.caption(f'"{c["text"].strip()[:240]}…"')
                        st.divider()

            if debug_info and debug_mode:
                with st.expander("🔍 Retrieval Debug"):
                    st.markdown(f"**Original Query:** `{debug_info.get('original_query')}`")
                    st.markdown(f"**Sub-Queries:** {debug_info.get('sub_queries_count', 1)}")
                    st.markdown("#### Query Understanding")
                    for sq in debug_info.get("sub_queries", []):
                        st.markdown(f"- `{sq.get('query')}` — Intent: `{sq.get('intent')}` | Type: `{sq.get('question_type')}`")
                    st.markdown("#### Retrieval Details")
                    st.json(debug_info.get("per_subquery_retrieval"))

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "isolated_citations": isolated_cits,
                "debug_info": debug_info,
            })


# =============================================================================
# TAB 2 — UPLOAD DOCUMENTS
# =============================================================================
with tab_upload:
    st.markdown("## 📤 Upload & Index Documents")
    st.markdown(
        "Drop any **PDF, DOCX, TXT, or Markdown** file below. "
        "It will be **automatically chunked, embedded, and added** to the search index "
        "without re-processing existing documents. "
        "Switch to the **💬 Chat** tab to immediately query it once indexing completes."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.info("⚡ **Incremental** — Only new docs are embedded.", icon="🔄")
    with c2:
        st.info("🛡️ **Duplicate-safe** — Identical files detected by SHA-256.", icon="🔍")
    with c3:
        st.info("📝 **Modified files** — Updated versions cleanly replace old chunks.", icon="✏️")

    st.markdown("---")

    up_file = st.file_uploader(
        "Choose a file to index",
        type=["pdf", "docx", "txt", "md"],
        key="main_doc_uploader",
        help="Automatically chunked, embedded, and indexed into FAISS + BM25.",
    )

    if up_file is not None:
        file_bytes = up_file.getvalue()
        fname      = up_file.name
        file_size  = len(file_bytes) / 1024
        st.markdown(f"**Selected:** `{fname}` ({file_size:.1f} KB)")

        status, status_info = check_document_status(fname, file_bytes)

        if status == "duplicate":
            chunk_count = status_info["registry_entry"].get("chunk_count", 0)
            st.warning(
                f"📄 **{fname}** is already indexed with **{chunk_count} chunks**. "
                "What would you like to do?"
            )
            col_skip, col_reidx = st.columns(2)
            with col_skip:
                if st.button("⏭️ Skip (already up to date)", key="btn_skip_tab", use_container_width=True):
                    st.toast(f"Skipped {fname} — already indexed.")
            with col_reidx:
                if st.button("🔄 Re-index anyway", key="btn_reidx_tab", type="primary", use_container_width=True):
                    with st.status(f"Re-indexing '{fname}'...", expanded=True) as s_box:
                        pb = st.progress(0.0)
                        def _prog_ri(msg, p): st.write(msg); pb.progress(p)
                        dest = os.path.join(config.DOCS_DIR, fname)
                        res  = ingest_new_document_incrementally(
                            file_path=dest, file_bytes=file_bytes,
                            vector_store=vector_store, progress_callback=_prog_ri, reindex=True,
                        )
                        s_box.update(label=f"✓ '{fname}' re-indexed!", state="complete", expanded=False)
                        st.success(f"🎉 **{fname}** re-indexed — {res['chunk_count']} chunks.")
                        reload_system_pipeline(); st.rerun()
        else:
            action_label = "Updating modified" if status == "modified" else "Ingesting new"
            if status == "modified":
                st.info(f"✏️ **{fname}** has changed — old chunks will be replaced.")
            with st.status(f"{action_label} document '{fname}'...", expanded=True) as s_box:
                pb = st.progress(0.0)
                def _prog_new(msg, p): st.write(msg); pb.progress(p)
                dest = os.path.join(config.DOCS_DIR, fname)
                try:
                    res = ingest_new_document_incrementally(
                        file_path=dest, file_bytes=file_bytes,
                        vector_store=vector_store, progress_callback=_prog_new,
                        reindex=(status == "modified"),
                    )
                    s_box.update(label=f"✓ '{fname}' is now searchable!", state="complete", expanded=False)
                    st.success(
                        f"🎉 **{fname}** indexed — {res['chunk_count']} chunks.\n\n"
                        "→ Switch to the **💬 Chat** tab to ask questions about it."
                    )
                    reload_system_pipeline(); st.rerun()
                except Exception as e:
                    s_box.update(label=f"❌ Failed to ingest '{fname}'", state="error", expanded=True)
                    st.error(f"Error: {e}")
    else:
        st.markdown("---")
        registry_now = load_document_registry()
        indexed_now  = {k: v for k, v in registry_now.items() if v.get("status") == "indexed"}
        if indexed_now:
            st.markdown(f"### Currently Indexed ({len(indexed_now)} document{'s' if len(indexed_now) != 1 else ''})")
            for doc_name, doc_info in indexed_now.items():
                cnt   = doc_info.get("chunk_count", 0)
                dtype = doc_info.get("document_type", "document").replace("_", " ").title()
                ts    = doc_info.get("indexed_at", "")[:19]
                st.markdown(f"- **{doc_name}** — {cnt} chunks · {dtype} · `{ts}`")
        else:
            st.info("No documents indexed yet. Upload your first document above!")


# =============================================================================
# TAB 3 — RAGAS EVALUATION
# =============================================================================
with tab_eval:
    st.markdown("## 📊 RAGAS + LangSmith Evaluation")
    st.markdown(
        "Measure your RAG pipeline quality with "
        "[RAGAS](https://docs.ragas.io) metrics. "
        "Results can be logged to [LangSmith](https://smith.langchain.com) for tracking."
    )

    langsmith_key = os.getenv("LANGSMITH_API_KEY", "").strip()
    if langsmith_key:
        project_name = os.getenv("LANGSMITH_PROJECT", "RAG-Assistant-Eval")
        st.success(f"🔗 LangSmith connected — runs log to project: `{project_name}`", icon="✅")
    else:
        st.warning(
            "⚠️ LangSmith not configured — add `LANGSMITH_API_KEY` to `.env` to enable "
            "experiment tracking. Evaluation still runs locally without it.",
            icon="🔑"
        )

    st.markdown("---")

    with st.expander("📖 What do these metrics measure?"):
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown("**🎯 Faithfulness** — Is the answer grounded in context? (1.0 = no hallucination)")
            st.markdown("**📌 Answer Relevancy** — Does the answer address the question? (1.0 = perfectly on-topic)")
        with col_m2:
            st.markdown("**🔬 Context Precision** — Are retrieved chunks useful? (high = low noise)")
            st.markdown("**📚 Context Recall** — Does context cover the full answer? (requires reference answer)")

    st.markdown("---")
    st.markdown("### ⚙️ Evaluation Setup")

    eval_mode = st.radio(
        "Question source:",
        ["Built-in test questions", "Enter custom questions", "Upload JSON eval set"],
        horizontal=True, key="eval_mode_radio",
    )

    DEFAULT_EVAL_QUESTIONS = [
        "What are my coursework subjects?",
        "How well am I doing academically overall?",
        "Where do I currently work?",
        "What model architecture did I use for emotion detection?",
        "Where did I study before university?",
    ]

    questions_to_eval = []
    reference_answers = []

    if eval_mode == "Built-in test questions":
        questions_to_eval = DEFAULT_EVAL_QUESTIONS
        st.info(f"Will evaluate {len(questions_to_eval)} built-in questions.")
        for i, q in enumerate(questions_to_eval, 1):
            st.caption(f"{i}. {q}")

    elif eval_mode == "Enter custom questions":
        raw_qs = st.text_area(
            "Enter one question per line:", height=140,
            placeholder="What is my CGPA?\nWhere do I work?\nWhat projects have I done?",
            key="eval_custom_qs",
        )
        if raw_qs.strip():
            questions_to_eval = [q.strip() for q in raw_qs.strip().splitlines() if q.strip()]
            st.caption(f"{len(questions_to_eval)} question(s).")
        raw_refs = st.text_area(
            "Optional reference answers (one per line, same order):", height=90,
            placeholder="8.0\nAirkrit\n...", key="eval_refs",
        )
        if raw_refs.strip():
            reference_answers = [r.strip() for r in raw_refs.strip().splitlines() if r.strip()]

    elif eval_mode == "Upload JSON eval set":
        up_eval = st.file_uploader(
            "Upload JSON eval set", type=["json"], key="eval_json_up",
            help='Format: [{"question": "...", "reference": "..."}, ...] or ["q1", "q2", ...]',
        )
        if up_eval is not None:
            try:
                loaded_eval = json.loads(up_eval.getvalue().decode("utf-8"))
                if loaded_eval and isinstance(loaded_eval[0], dict):
                    questions_to_eval = [q.get("question", "") for q in loaded_eval if q.get("question")]
                    reference_answers = [q.get("reference", "") for q in loaded_eval]
                else:
                    questions_to_eval = [q for q in loaded_eval if q]
                st.success(f"Loaded {len(questions_to_eval)} questions.")
            except Exception as e:
                st.error(f"JSON parse error: {e}")

    st.markdown("### 📐 Metrics to Run")
    cma, cmb, cmc, cmd = st.columns(4)
    with cma: run_faith  = st.checkbox("🎯 Faithfulness",      value=True,  key="chk_faith")
    with cmb: run_relev  = st.checkbox("📌 Answer Relevancy",  value=True,  key="chk_relev")
    with cmc: run_prec   = st.checkbox("🔬 Context Precision", value=True,  key="chk_prec")
    with cmd: run_recall = st.checkbox("📚 Context Recall",    value=False, key="chk_recall",
                                        help="Needs reference answers.")

    selected_metrics = []
    if run_faith:  selected_metrics.append("faithfulness")
    if run_relev:  selected_metrics.append("answer_relevancy")
    if run_prec:   selected_metrics.append("context_precision")
    if run_recall: selected_metrics.append("context_recall")

    enable_ls = st.toggle(
        "🔗 Log results to LangSmith",
        value=bool(langsmith_key), disabled=not bool(langsmith_key),
        key="toggle_ls",
    )

    st.markdown("---")

    can_run = bool(questions_to_eval) and bool(selected_metrics) and vector_store is not None
    if not questions_to_eval:
        st.info("👆 Select or enter questions above to enable evaluation.")
    elif not selected_metrics:
        st.warning("Select at least one metric above.")
    elif vector_store is None:
        st.error("Vector store not loaded — upload and index documents first.")

    run_btn = st.button(
        "▶️  Run RAGAS Evaluation", type="primary",
        disabled=not can_run, key="btn_run_ragas",
    )

    if run_btn and can_run:
        st.session_state["ragas_results"] = None
        if enable_ls and langsmith_key:
            try:
                from ragas_eval import setup_langsmith_tracing
                setup_langsmith_tracing()
            except Exception as e:
                st.warning(f"LangSmith setup warning: {e}")

        with st.status("Running RAGAS evaluation…", expanded=True) as eval_status:
            prog_bar = st.progress(0.0)
            prog_ph  = st.empty()
            def _eval_prog(msg, p):
                prog_ph.write(f"→ {msg}")
                prog_bar.progress(min(float(p), 1.0))
            try:
                from ragas_eval import run_ragas_evaluation
                results = run_ragas_evaluation(
                    questions=questions_to_eval,
                    reference_answers=reference_answers if reference_answers else None,
                    vector_store=vector_store,
                    metrics_to_run=selected_metrics,
                    progress_callback=_eval_prog,
                )
                st.session_state["ragas_results"] = results
                eval_status.update(label="✓ RAGAS evaluation complete!", state="complete", expanded=False)
            except Exception as e:
                import traceback
                eval_status.update(label="❌ Evaluation failed", state="error", expanded=True)
                st.error(f"Error: {e}")
                st.code(traceback.format_exc())

    # -- Display results --
    if st.session_state.get("ragas_results"):
        res = st.session_state["ragas_results"]
        if res.get("error") and not res.get("scores"):
            st.error(f"Evaluation error: {res['error']}")
        else:
            st.markdown("---")
            st.markdown("### 📈 Evaluation Results")
            ts = res.get("timestamp", "")[:19]
            st.caption(f"Evaluated {len(res.get('per_question', []))} questions · {ts}")

            scores = res.get("scores", {})
            ICONS = {"faithfulness": "🎯", "answer_relevancy": "📌",
                     "context_precision": "🔬", "context_recall": "📚"}
            if scores:
                score_cols = st.columns(max(len(scores), 1))
                for i, (mn, score) in enumerate(scores.items()):
                    with score_cols[i]:
                        icon  = ICONS.get(mn, "📊")
                        label = mn.replace("_", " ").title()
                        if score is None:
                            st.metric(f"{icon} {label}", "N/A")
                        else:
                            badge = "✅" if score >= 0.7 else ("⚠️" if score >= 0.4 else "❌")
                            st.metric(f"{icon} {label}", f"{score:.3f}  {badge}")

            g1, g2, g3 = st.columns(3)
            with g1: st.success("≥ 0.70 — Good")
            with g2: st.warning("0.40–0.69 — Needs work")
            with g3: st.error("< 0.40 — Poor")

            st.markdown("---")
            st.markdown("#### 📋 Per-Question Breakdown")
            per_q = res.get("per_question", [])
            if per_q:
                mns = [m for m in selected_metrics if m in (per_q[0] if per_q else {})]
                for i, row in enumerate(per_q, 1):
                    q_txt = row.get("question", f"Q{i}")
                    with st.expander(f"Q{i}: {q_txt[:90]}{'...' if len(q_txt) > 90 else ''}"):
                        st.markdown(f"**Answer:** {str(row.get('answer', ''))[:400]}")
                        if mns:
                            sc_cols = st.columns(len(mns))
                            for j, mn in enumerate(mns):
                                val = row.get(mn)
                                with sc_cols[j]:
                                    if val is None or (isinstance(val, float) and math.isnan(val)):
                                        st.metric(mn.replace("_", " ").title(), "\u2014")
                                    else:
                                        badge = "\u2705" if val >= 0.7 else ("\u26a0\ufe0f" if val >= 0.4 else "\u274c")
                                        st.metric(mn.replace("_", " ").title(), f"{val:.3f} {badge}")

            if res.get("errors"):
                with st.expander(f"\u26a0\ufe0f Errors ({len(res['errors'])})"):
                    for err in res["errors"]:
                        st.json(err)

            st.markdown("---")
            st.download_button(
                "\u2b07\ufe0f  Download Full Results (JSON)",
                data=json.dumps(res, indent=2),
                file_name=f"ragas_results_{res.get('timestamp', 'run')[:10]}.json",
                mime="application/json",
            )
            if enable_ls and langsmith_key:
                proj = os.getenv("LANGSMITH_PROJECT", "RAG-Assistant-Eval")
                st.info(
                    f"\ud83d\udcca Results logged to LangSmith \u2192 "
                    f"[View project](https://smith.langchain.com/o/me/projects/{proj})",
                    icon="\ud83d\udd17"
                )

