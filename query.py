"""
Query pipeline with optional metadata filtering by source file.

Solves cross-document ambiguity: when two documents share overlapping
vocabulary (e.g. both mention "course"/"coursework"), pure vector search
can retrieve from the wrong document. Filtering by source forces retrieval
to stay within the document you actually mean.

Run (searches all documents):
    python query.py "What is my CGPA?"

Run (searches only one file):
    python query.py "What are my courses?" --source "resume with work.pdf"
"""

import sys
import os
import argparse
from dotenv import load_dotenv

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

import config

PROMPT_TEMPLATE = """You are a helpful assistant answering questions using
ONLY the context provided below. If the answer isn't in the context, say
"I don't have enough information to answer that" — do not make things up.

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


def retrieve(vector_store, question, k=config.TOP_K, source_filter=None):
    """
    Retrieve top-k chunks. If source_filter is given, only chunks whose
    metadata 'source' contains that string are eligible - this sidesteps
    cases where vector similarity alone picks the wrong document.
    """
    fetch_k = k * 5 if source_filter else k
    results = vector_store.similarity_search_with_score(question, k=fetch_k)

    if source_filter:
        results = [
            (doc, score) for doc, score in results
            if source_filter.lower() in doc.metadata.get("source", "").lower()
        ]

    results = results[:k]

    for i, (doc, score) in enumerate(results, 1):
        source = doc.metadata.get("source", "unknown")
        print(f"  [{i}] score={score:.3f} | {source}")

    return [doc for doc, _ in results]


def generate_answer(question, retrieved_docs):
    if not retrieved_docs:
        return "I don't have enough information to answer that."

    context = "\n\n---\n\n".join(doc.page_content for doc in retrieved_docs)
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    api_key = os.getenv("GROQ_API_KEY")
    model_name = getattr(config, "GROQ_MODEL", "openai/gpt-oss-20b")
    llm = ChatGroq(model=model_name, temperature=0, api_key=api_key)
    chain = prompt | llm
    response = chain.invoke({"context": context, "question": question})
    return response.content



def answer_question(question, source_filter=None):
    vector_store = load_vector_store()
    label = f" (filtered to '{source_filter}')" if source_filter else ""
    print(f"\nRetrieving top {config.TOP_K} chunks{label}...\n")
    docs = retrieve(vector_store, question, source_filter=source_filter)
    answer = generate_answer(question, docs)
    print(f"\nAnswer:\n{answer}\n")
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