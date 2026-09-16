"""
Debug tool — dump all chunks from a specific source file so you can see
exactly what text got extracted and how it was split.

Run:
    python debug_chunks.py "sem6.pdf"
"""

import sys
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

import config


def dump_chunks_for_source(filename_filter):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.load_local(
        config.VECTOR_STORE_DIR,
        embeddings,
        allow_dangerous_deserialization=True,
    )

    # Pull every chunk out of the FAISS docstore
    all_docs = list(vector_store.docstore._dict.values())
    matching = [d for d in all_docs if filename_filter in d.metadata.get("source", "")]

    print(f"Found {len(matching)} chunks from files matching '{filename_filter}':\n")
    for i, doc in enumerate(matching, 1):
        print(f"--- Chunk {i} (source: {doc.metadata.get('source')}) ---")
        print(doc.page_content)
        print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python debug_chunks.py "sem6.pdf"')
        sys.exit(1)
    dump_chunks_for_source(sys.argv[1])
