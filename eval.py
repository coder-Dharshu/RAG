"""
Eval harness — Phase 4.

A small, hand-written set of question/expected-answer pairs pulled from
your own documents (things you know the ground truth for). Running this
after any pipeline change (chunk size, hybrid search, reranking, etc.)
gives you a pass/fail score instead of manually re-testing each question.

Two things get checked per question:
  1. Retrieval hit: did any of the retrieved chunks come from the
     expected source file? (catches cross-document ambiguity, like the
     "coursework" bug)
  2. Answer match: does the generated answer contain the expected
     keyword(s)? (catches dilution/generation failures, like the
     registration number bug)

Run:
    python eval.py
"""

import sys
import os
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["HF_HUB_OFFLINE"] = "1"
import time
import re
import json
import argparse
import warnings
warnings.filterwarnings("ignore")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from query import load_vector_store, retrieve, generate_answer
import config

# --- Edit this list to match YOUR documents ---
# Each entry: question, expected_source (substring match), expected_keywords
# (all must appear, case-insensitive, in the generated answer)
#
# These are deliberately harder than a first-pass eval set: ambiguous
# phrasing, no source hints, cross-document competition, and multi-fact
# answers — designed to actually stress the retriever instead of
# confirming what we already know works.
EVAL_SET = [
    {
        # Ambiguous: "coursework" collides with sem6.pdf's course table.
        # No --source hint given — this is the exact bug we found manually.
        "question": "What are my coursework subjects?",
        "expected_source": "resume with work.pdf",
        "expected_keywords": ["Data Structures", "Operating Systems"],
    },
    {
        # Vague phrasing, no exact keyword match to "CGPA" in the question.
        "question": "How well am I doing academically overall?",
        "expected_source": "resume with work.pdf",
        "expected_keywords": ["8.0"],
    },
    {
        # Multi-hop: requires pulling BOTH semester grade point AND overall
        # CGPA, which live in two different documents.
        "question": "What's the difference between my CGPA and this semester's SGPA?",
        "expected_source": "sem6.pdf",
        "expected_keywords": ["8.3"],
    },
    {
        # Requires aggregating across multiple table rows, not a single fact.
        "question": "Which courses did I get an A+ grade in?",
        "expected_source": "sem6.pdf",
        "expected_keywords": ["Neural Networks"],
    },
    {
        # "Score" is ambiguous — could mean CGPA, SGPA, or an individual
        # course mark. Tests whether retrieval grabs the right one.
        "question": "What score did I get in Optimization Techniques?",
        "expected_source": "sem6.pdf",
        "expected_keywords": ["77"],
    },
    {
        # Cross-document term collision: "experience" appears in both the
        # internship section AND could loosely relate to project descriptions.
        "question": "Where do I currently work?",
        "expected_source": "resume with work.pdf",
        "expected_keywords": ["Airkrit"],
    },
    {
        # Requires connecting a project NAME to its TECH STACK — two
        # separate facts that must both be retrieved and combined.
        "question": "What model architecture did I use for emotion detection, and what dataset?",
        "expected_source": "resume with work.pdf",
        "expected_keywords": ["MobileNet", "FER-2013"],
    },
    {
        # No document filename or field name mentioned at all — fully
        # natural phrasing a real user would type.
        "question": "Where did I study before university?",
        "expected_source": "resume with work.pdf",
        "expected_keywords": ["National College"],
    },
    {
        # Negative test: this fact does NOT exist in any document.
        # A good system should say it doesn't know — NOT hallucinate.
        "question": "What is my father's occupation?",
        "expected_source": None,  # no valid source should "win" this
        "expected_keywords": ["don't have enough information"],
    },
    {
        # Distractor test: sample.txt (unrelated company policy doc) sits
        # in the same vector store — this checks it doesn't leak into an
        # answer about the user's own background.
        "question": "What is my refund policy?",
        "expected_source": None,
        "expected_keywords": ["don't have enough information"],
    },
]


def check_retrieval_hit(retrieved_docs, expected_source):
    """
    True if at least one retrieved chunk came from the expected file.
    If expected_source is None, this is a negative/distractor test (the
    fact shouldn't exist anywhere) — retrieval isn't the thing being
    checked, so it always passes and the answer check does the real work.
    """
    if expected_source is None:
        return True
    return any(
        expected_source.lower() in doc.metadata.get("source", "").lower()
        for doc in retrieved_docs
    )


def check_answer_match(answer, expected_keywords):
    """
    True if every expected keyword appears in the answer (case-insensitive & hyphen-normalized).
    Normalizes common Unicode punctuation variants (typographic hyphens,
    en/em dashes) to plain ASCII first — otherwise a correct answer can
    fail the check just because the LLM generated 'FER‑2013' (U+2011)
    instead of 'FER-2013' (ASCII hyphen).
    """
    def normalize(text):
        import unicodedata
        text = unicodedata.normalize("NFKC", text)
        text = re.sub(r'[\u2010-\u2015\u2212]', '-', text.lower())
        text = re.sub(r'\s+', ' ', text)
        return text

    answer_norm = normalize(answer)
    return all(normalize(kw) in answer_norm for kw in expected_keywords)


def run_eval(eval_set=None):
    eval_set = eval_set or EVAL_SET
    vector_store = load_vector_store()

    retrieval_hits = 0
    answer_hits = 0
    results = []

    for i, case in enumerate(eval_set, 1):
        question = case.get("question", "")
        if not question:
            continue
        docs = retrieve(vector_store, question, k=config.TOP_K)
        answer = generate_answer(question, docs)

        expected_source = case.get("expected_source")
        expected_keywords = case.get("expected_keywords", [])

        retrieval_ok = check_retrieval_hit(docs, expected_source)
        answer_ok = check_answer_match(answer, expected_keywords)

        retrieval_hits += retrieval_ok
        answer_hits += answer_ok

        status = "PASS" if (retrieval_ok and answer_ok) else "FAIL"
        results.append({
            "n": i,
            "question": question,
            "status": status,
            "retrieval_ok": retrieval_ok,
            "answer_ok": answer_ok,
            "answer": answer,
        })

        print(f"[{status}] Q{i}: {question}", flush=True)
        if not retrieval_ok:
            print(f"    ✗ Retrieval miss — expected source: {expected_source}", flush=True)
        if not answer_ok:
            print(f"    ✗ Answer missing expected keyword(s): {expected_keywords}", flush=True)
            print(f"    → Got: {answer.strip().splitlines()[0] if answer.strip() else '(empty)'}", flush=True)
        print(flush=True)
        time.sleep(0.5)

    total = len(results)
    print("=" * 50, flush=True)
    print(f"Retrieval accuracy: {retrieval_hits}/{total} ({100*retrieval_hits/max(total,1):.0f}%)", flush=True)
    print(f"Answer accuracy:    {answer_hits}/{total} ({100*answer_hits/max(total,1):.0f}%)", flush=True)
    print(f"Full pass:          {sum(1 for r in results if r['status']=='PASS')}/{total}", flush=True)
    print("=" * 50, flush=True)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", default=None,
                         help="Path to a JSON eval set generated by generate_eval_questions.py. "
                              "If omitted, uses the hand-written EVAL_SET in this file.")
    parser.add_argument("--limit", type=int, default=None,
                         help="Limit the number of questions to evaluate (for quick spot-checks).")
    parser.add_argument("--offset", type=int, default=0,
                         help="Starting index offset for evaluation questions.")
    args = parser.parse_args()

    eval_data = None
    if args.set:
        with open(args.set, "r", encoding="utf-8") as f:
            eval_data = json.load(f)
        print(f"Loaded {len(eval_data)} questions from '{args.set}'\n", flush=True)
    else:
        eval_data = EVAL_SET

    if eval_data:
        start = args.offset
        end = start + args.limit if args.limit is not None else len(eval_data)
        eval_data = eval_data[start:end]
        print(f"Evaluating {len(eval_data)} question(s) (offset {start})...\n", flush=True)

    run_eval(eval_data)