"""
Phase 23 — Comprehensive End-to-End Test Suite.

Runs all 10 test cases specified in the requirements:
TEST 1: "What is my CGPA?"
TEST 2: "What are my projects?"
TEST 3: "List all my projects."
TEST 4: "What technologies are used in my projects?"
TEST 5: "Which of my projects use Python?"
TEST 6: "Tell me about my Intelligent Document Processing Platform."
TEST 7: "What tools were used in the quantum Bloch sphere research paper?"
TEST 8: "What is my CGPA and what are my projects?"
TEST 9: "What is my CGPA, what are my projects, and what tools were used in the quantum Bloch sphere paper?"
TEST 10: "What projects have I done involving AI?"
"""

from __future__ import annotations

import sys
import os
import json
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from query import run_rag_pipeline, load_vector_store

TEST_CASES = [
    {
        "id": "TEST 1",
        "query": "What is my CGPA?",
        "expected_check": lambda ans, cits: ("8.0" in ans or "8.3" in ans) and any("resume" in c.lower() or "sem6" in c.lower() for sq_cits in cits.values() for c in [x["filename"] for x in sq_cits]),
    },
    {
        "id": "TEST 2",
        "query": "What are my projects?",
        "expected_check": lambda ans, cits: "Intelligent Document Processing" in ans and "Facial Emotion Recognition" in ans and "Fin-Agent" in ans,
    },
    {
        "id": "TEST 3",
        "query": "List all my projects.",
        "expected_check": lambda ans, cits: "Intelligent Document Processing" in ans and "Facial Emotion Recognition" in ans and "Fin-Agent" in ans,
    },
    {
        "id": "TEST 4",
        "query": "What technologies are used in my projects?",
        "expected_check": lambda ans, cits: "Python" in ans and ("FastAPI" in ans or "Docker" in ans or "React" in ans or "Groq" in ans),
    },
    {
        "id": "TEST 5",
        "query": "Which of my projects use Python?",
        "expected_check": lambda ans, cits: "Intelligent Document Processing" in ans and "Facial Emotion Recognition" in ans and "Fin-Agent" not in ans,
    },
    {
        "id": "TEST 6",
        "query": "Tell me about my Intelligent Document Processing Platform.",
        "expected_check": lambda ans, cits: "FastAPI" in ans or "Qwen" in ans or "microservices" in ans,
    },
    {
        "id": "TEST 7",
        "query": "What tools were used in the quantum Bloch sphere research paper?",
        "expected_check": lambda ans, cits: "FPGA" in ans or "OpenCL" in ans,
    },
    {
        "id": "TEST 8",
        "query": "What is my CGPA and what are my projects?",
        "expected_check": lambda ans, cits: "8.0" in ans and "Intelligent Document Processing" in ans and "Fin-Agent" in ans,
    },
    {
        "id": "TEST 9",
        "query": "What is my CGPA, what are my projects, and what tools were used in the quantum Bloch sphere paper?",
        "expected_check": lambda ans, cits: len(cits) >= 3,
    },
    {
        "id": "TEST 10",
        "query": "What projects have I done involving AI?",
        "expected_check": lambda ans, cits: ("Intelligent Document Processing" in ans or "Facial Emotion Recognition" in ans or "Fin-Agent" in ans) and not any("The_interactive_system" in x["filename"] for sq, cs in cits.items() if "project" in sq.lower() for x in cs),
    },
]


def run_tests():
    print(f"\n{'='*70}")
    print("STARTING COMPREHENSIVE 10-POINT RAG TEST SUITE")
    print(f"{'='*70}\n")

    vector_store = load_vector_store()
    results_summary = []

    for test in TEST_CASES:
        tid = test["id"]
        q = test["query"]
        print(f"\n{'-'*60}")
        print(f"RUNNING {tid}: \"{q}\"")
        print(f"{'-'*60}")

        start_time = time.time()
        res = run_rag_pipeline(q, vector_store=vector_store, debug=True)
        elapsed = time.time() - start_time

        ans = res["answer"]
        cits = res["isolated_citations"]

        passed = test["expected_check"](ans, cits)

        print(f"\n[Status]: {'✅ PASSED' if passed else '❌ FAILED'} in {elapsed:.2f}s")
        print("\nAnswer Snippet:")
        print(ans[:450] + ("..." if len(ans) > 450 else ""))

        print("\nIsolated Sources:")
        for sq, sq_cits in cits.items():
            sources_str = ", ".join(set(c["filename"] for c in sq_cits))
            print(f"  • \"{sq}\" -> [{sources_str}]")

        results_summary.append({
            "id": tid,
            "query": q,
            "passed": passed,
            "elapsed_seconds": round(elapsed, 2),
            "sources_by_subquery": {sq: list(set(c["filename"] for c in sq_cits)) for sq, sq_cits in cits.items()}
        })

    print(f"\n\n{'='*70}")
    print("FINAL TEST RESULTS SUMMARY")
    print(f"{'='*70}")
    all_passed = True
    for r in results_summary:
        status_icon = "✅ PASS" if r["passed"] else "❌ FAIL"
        if not r["passed"]:
            all_passed = False
        print(f"{r['id']:8} | {status_icon} | {r['elapsed_seconds']}s | {r['query']}")

    print(f"\nOVERALL RESULT: {'ALL TESTS PASSED! 🎉' if all_passed else 'SOME TESTS FAILED'}\n")

    with open("test_results.json", "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2)


if __name__ == "__main__":
    run_tests()
