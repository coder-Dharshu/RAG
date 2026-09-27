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
import re
import json
import argparse

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
    True if every expected keyword is semantically present in the answer.

    Handles common false-failure cases:
    1. Unicode punctuation (typographic hyphens, superscripts, degree symbols)
    2. Whitespace differences ("θ=180°" vs "θ = 180°")
    3. Partial phrase match — if the keyword is multi-word, ALL individual
       content words must appear somewhere in the answer, even if not
       adjacent (catches paraphrased but correct answers like
       "two microcontrollers" → answer says "Two ... microcontrollers")
    """
    def normalize(text):
        # Unicode dashes/hyphens → ASCII hyphen
        for dash in ["\u2010", "\u2011", "\u2012", "\u2013", "\u2014", "\u2212"]:
            text = text.replace(dash, "-")
        # Unicode superscripts → plain digits
        superscript_map = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")
        text = text.translate(superscript_map)
        # Collapse all whitespace so "θ = 180°" matches "θ=180°"
        import re
        text = re.sub(r'\s+', ' ', text)
        return text.lower()

    def keyword_present(answer_norm, kw_norm):
        # Try exact substring match first
        if kw_norm in answer_norm:
            return True
        # Fall back to "all content words present" match
        # (handles paraphrasing like "Two" for "two microcontrollers")
        stop_words = {"the", "a", "an", "of", "in", "is", "are", "and",
                      "to", "for", "on", "at", "as", "by", "or", "its"}
        content_words = [
            w for w in kw_norm.split()
            if w not in stop_words and len(w) > 2
        ]
        if not content_words:
            return kw_norm in answer_norm
        return all(w in answer_norm for w in content_words)

    answer_norm = normalize(answer)
    return all(keyword_present(answer_norm, normalize(kw)) for kw in expected_keywords)


def categorize_question(question):
    """
    Rough auto-categorization so a 100+ question eval run shows WHERE
    it's failing, not just an aggregate score that hides the pattern.
    """
    q = question.lower()
    if any(w in q for w in ["figure", "fig.", "shown in", "diagram", "image"]):
        return "figure-reference"
    if any(w in q for w in ["equation", "formula", "θ", "φ", "calculate", "angle"]):
        return "math/formula"
    if any(w in q for w in ["acronym", "stand for", "fpga", "opencl", "ble", "iot"]):
        return "acronym/technical-term"
    if any(w in q for w in ["reference", "cited", "author", "published"]):
        return "citation/metadata"
    return "general-fact"


def run_eval(eval_set=None):
    eval_set = eval_set or EVAL_SET
    vector_store = load_vector_store()

    retrieval_hits = 0
    answer_hits = 0
    results = []

    for i, case in enumerate(eval_set, 1):
        question = case["question"]
        if not question:
            continue
        docs = retrieve(vector_store, question, k=config.TOP_K)
        answer = generate_answer(question, docs)

        retrieval_ok = check_retrieval_hit(docs, case["expected_source"])
        answer_ok = check_answer_match(answer, case["expected_keywords"])

        retrieval_hits += retrieval_ok
        answer_hits += answer_ok

        status = "PASS" if (retrieval_ok and answer_ok) else "FAIL"
        results.append({
            "n": i,
            "question": question,
            "category": categorize_question(question),
            "status": status,
            "retrieval_ok": retrieval_ok,
            "answer_ok": answer_ok,
            "answer": answer,
        })

        print(f"[{status}] Q{i}: {question}")
        if not retrieval_ok:
            print(f"    ✗ Retrieval miss — expected source: {case['expected_source']}")
        if not answer_ok:
            print(f"    ✗ Answer missing expected keyword(s): {case['expected_keywords']}")
            print(f"    → Got: {answer[:150]}")
        print()

    total = len(results)
    print("=" * 50)
    print(f"Retrieval accuracy: {retrieval_hits}/{total} ({100*retrieval_hits/max(total,1):.0f}%)")
    print(f"Answer accuracy:    {answer_hits}/{total} ({100*answer_hits/max(total,1):.0f}%)")
    print(f"Full pass:          {sum(1 for r in results if r['status']=='PASS')}/{total}")

    # Per-category breakdown — this is what actually tells you where to
    # focus next, since an aggregate score hides which question TYPES
    # are driving failures.
    print("\nBy category:")
    categories = sorted(set(r["category"] for r in results))
    for cat in categories:
        cat_results = [r for r in results if r["category"] == cat]
        cat_pass = sum(1 for r in cat_results if r["status"] == "PASS")
        print(f"  {cat:<25} {cat_pass}/{len(cat_results)} "
              f"({100*cat_pass/len(cat_results):.0f}%)")
    print("=" * 50)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", default=None,
                         help="Path to a JSON eval set generated by generate_eval_questions.py. "
                              "If omitted, uses the hand-written EVAL_SET in this file.")
    args = parser.parse_args()

    if args.set:
        with open(args.set, "r", encoding="utf-8") as f:
            loaded_set = json.load(f)
        print(f"Loaded {len(loaded_set)} questions from '{args.set}'\n")
        run_eval(loaded_set)
    else:
        run_eval()