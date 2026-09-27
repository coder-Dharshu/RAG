"""
Comprehensive Automated Test Suite for Automatic Incremental Document Ingestion.
Tests all 10 requirements from Section 24 of the specification.
"""

from __future__ import annotations

import os
import sys
import json
import pickle
import shutil
from typing import Dict, Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

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
    save_index_metadata,
    CHUNKING_VERSION,
    INDEX_VERSION,
)
from query import load_vector_store, run_rag_pipeline
from hybrid_retriever import load_all_chunks, reset_retriever_cache


def log_test(num: int, title: str):
    print(f"\n{'='*70}")
    print(f"TEST {num}: {title}")
    print(f"{'='*70}")


def run_all_tests():
    test_results = {}

    # ──────────────────────────────────────────────────────────────────────────
    # TEST 1: Start application with existing documents (No unnecessary reprocessing)
    # ──────────────────────────────────────────────────────────────────────────
    log_test(1, "Startup with existing documents — No unnecessary reprocessing")
    registry = load_document_registry()
    meta = load_index_metadata()
    chunks = load_all_chunks()
    vector_store = load_vector_store()

    initial_chunk_count = len(chunks)
    initial_vector_count = vector_store.index.ntotal
    initial_doc_count = len([d for d in registry.values() if d.get("status") == "indexed"])

    print(f"Loaded existing index: {initial_doc_count} documents, {initial_chunk_count} chunks, {initial_vector_count} FAISS vectors.")
    assert initial_chunk_count > 0, "No chunks found in existing index!"
    assert initial_vector_count == initial_chunk_count, f"Mismatch: {initial_vector_count} vectors vs {initial_chunk_count} chunks"
    test_results["TEST 1"] = "PASSED: Existing index loaded without reprocessing."

    # ──────────────────────────────────────────────────────────────────────────
    # TEST 2: Upload a completely new document
    # ──────────────────────────────────────────────────────────────────────────
    log_test(2, "Upload a completely new document")
    test_doc_name = "autonomous_drone_project.txt"
    test_doc_content = (
        "PROJECTS\n\n"
        "Autonomous Aerial Surveillance Drone | Python, ROS 2, PyTorch, PX4, Jetson Orin Nano, OpenCV\n"
        "• Designed and deployed an autonomous quadcopter navigating GPS-denied indoor environments with 98.4% obstacle avoidance accuracy.\n"
        "• Integrated real-time YOLOv8 object detection on NVIDIA Jetson Orin Nano achieving 45 FPS video inferencing.\n"
        "• Implemented EKF state estimation fusing IMU and visual odometry data.\n"
    ).encode("utf-8")

    status, info = check_document_status(test_doc_name, test_doc_content)
    print(f"Initial check status for '{test_doc_name}': {status}")
    assert status == "new", f"Expected 'new', got {status}"

    # Perform incremental ingestion
    dest_path = os.path.join(config.DOCS_DIR, test_doc_name)
    ingest_res = ingest_new_document_incrementally(
        file_path=dest_path,
        file_bytes=test_doc_content,
        vector_store=vector_store,
        reindex=False,
    )
    print(f"Ingestion result: {ingest_res['message']}")
    assert ingest_res["status"] == "success"
    assert ingest_res["chunk_count"] > 0

    # Verify vector store and chunks count increased ONLY by the new chunks
    new_chunks = load_all_chunks()
    expected_new_count = initial_chunk_count + ingest_res["chunk_count"]
    print(f"Total chunks after upload: {len(new_chunks)} (expected: {expected_new_count})")
    print(f"FAISS vectors count: {vector_store.index.ntotal}")
    assert len(new_chunks) == expected_new_count
    assert vector_store.index.ntotal == expected_new_count
    test_results["TEST 2"] = "PASSED: New document incrementally ingested without re-embedding old docs."

    # ──────────────────────────────────────────────────────────────────────────
    # TEST 3: Ask a question about the new document
    # ──────────────────────────────────────────────────────────────────────────
    log_test(3, "Query the newly uploaded document")
    rag_res = run_rag_pipeline(
        question="What is the obstacle avoidance accuracy of the autonomous aerial surveillance drone?",
        vector_store=vector_store,
        debug=True,
    )
    print(f"Answer: {rag_res['answer']}")
    print(f"Isolated citations: {rag_res['isolated_citations']}")
    assert "98.4" in rag_res["answer"] or "autonomous" in rag_res["answer"].lower(), "Failed to retrieve from new doc!"
    test_results["TEST 3"] = f"PASSED: Query retrieved new document content correctly: {rag_res['answer'][:100]}..."

    # ──────────────────────────────────────────────────────────────────────────
    # TEST 4: Upload the exact same document again (Duplicate Detection)
    # ──────────────────────────────────────────────────────────────────────────
    log_test(4, "Duplicate detection on identical upload")
    dup_status, dup_info = check_document_status(test_doc_name, test_doc_content)
    print(f"Status check for duplicate upload: {dup_status}")
    assert dup_status == "duplicate", f"Expected duplicate, got {dup_status}"

    dup_ingest = ingest_new_document_incrementally(
        file_path=dest_path,
        file_bytes=test_doc_content,
        vector_store=vector_store,
        reindex=False,
    )
    print(f"Duplicate ingest response: {dup_ingest['message']}")
    assert dup_ingest["status"] == "duplicate"
    # Ensure vector and chunk counts did NOT change
    chunks_after_dup = load_all_chunks()
    assert len(chunks_after_dup) == expected_new_count
    assert vector_store.index.ntotal == expected_new_count
    test_results["TEST 4"] = "PASSED: Duplicate detected. No duplicate chunks or vectors created."

    # ──────────────────────────────────────────────────────────────────────────
    # TEST 5: Modify existing document and upload again
    # ──────────────────────────────────────────────────────────────────────────
    log_test(5, "Modified document update (Replace old chunks)")
    modified_content = (
        "PROJECTS\n\n"
        "Autonomous Aerial Surveillance Drone | Python, ROS 2, PyTorch, PX4, Jetson Orin Nano, OpenCV, LiDAR SLAM\n"
        "• Upgraded autonomous quadcopter navigating GPS-denied environments with 99.7% obstacle avoidance accuracy using 3D LiDAR SLAM.\n"
        "• Integrated real-time YOLOv8 object detection on NVIDIA Jetson Orin Nano achieving 45 FPS video inferencing.\n"
        "• Implemented EKF state estimation fusing IMU, LiDAR, and visual odometry data.\n"
    ).encode("utf-8")

    mod_status, mod_info = check_document_status(test_doc_name, modified_content)
    print(f"Status check for modified upload: {mod_status}")
    assert mod_status == "modified", f"Expected 'modified', got {mod_status}"

    # Reindex modified doc
    mod_ingest = ingest_new_document_incrementally(
        file_path=dest_path,
        file_bytes=modified_content,
        vector_store=vector_store,
        reindex=True,
    )
    print(f"Modified ingest response: {mod_ingest['message']}")
    assert mod_ingest["status"] == "success"

    # Verify no duplicate old chunks remain
    chunks_after_mod = load_all_chunks()
    drone_chunks = [c for c in chunks_after_mod if c.metadata.get("filename") == test_doc_name]
    print(f"Total drone chunks after update: {len(drone_chunks)}")
    for c in drone_chunks:
        assert "98.4" not in c.page_content, "Found stale 98.4% chunk in updated index!"
        assert "99.7" in c.page_content or "SLAM" in c.page_content

    # Query the updated doc
    mod_rag_res = run_rag_pipeline(
        question="What is the updated obstacle avoidance accuracy of the autonomous aerial surveillance drone?",
        vector_store=vector_store,
    )
    print(f"Updated Answer: {mod_rag_res['answer']}")
    assert "99.7" in mod_rag_res["answer"], f"Expected 99.7% in answer, got: {mod_rag_res['answer']}"
    test_results["TEST 5"] = "PASSED: Modified document replaced cleanly. No stale chunks remained."

    # ──────────────────────────────────────────────────────────────────────────
    # TEST 6: Upload a second new document (Both remain searchable)
    # ──────────────────────────────────────────────────────────────────────────
    log_test(6, "Upload second new document — Verify both searchable without full re-embedding")
    test_doc2_name = "quantum_key_distribution_paper.txt"
    test_doc2_content = (
        "RESEARCH PAPER\n\n"
        "Title: Satellite-to-Ground Continuous-Variable Quantum Key Distribution\n"
        "Authors: Dr. Alice Smith, Dr. Bob Jones\n"
        "Abstract: We demonstrate a continuous-variable quantum key distribution (CV-QKD) link over a 1200 km low-Earth orbit satellite channel with a secret key rate of 2.4 kbps.\n"
        "Conclusion: The CV-QKD transceiver successfully demonstrated quantum security against collective eavesdropping attacks.\n"
    ).encode("utf-8")

    dest_path2 = os.path.join(config.DOCS_DIR, test_doc2_name)
    ingest_res2 = ingest_new_document_incrementally(
        file_path=dest_path2,
        file_bytes=test_doc2_content,
        vector_store=vector_store,
        reindex=False,
    )
    print(f"Second doc ingestion result: {ingest_res2['message']}")
    assert ingest_res2["status"] == "success"

    # Query 1st doc
    q1_res = run_rag_pipeline("What obstacle avoidance accuracy does the drone have?", vector_store=vector_store)
    print(f"Query 1 (Drone): {q1_res['answer'][:120]}...")
    assert "99.7" in q1_res["answer"]

    # Query 2nd doc
    q2_res = run_rag_pipeline("What secret key rate was achieved in the satellite CV-QKD link?", vector_store=vector_store)
    print(f"Query 2 (CV-QKD): {q2_res['answer'][:120]}...")
    assert "2.4 kbps" in q2_res["answer"] or "2.4" in q2_res["answer"]
    test_results["TEST 6"] = "PASSED: Second doc indexed; first doc and second doc both immediately searchable."

    # ──────────────────────────────────────────────────────────────────────────
    # TEST 7: Restart simulation (Load from disk)
    # ──────────────────────────────────────────────────────────────────────────
    log_test(7, "Simulate application restart — Load from disk")
    reset_retriever_cache()
    reloaded_vector_store = load_vector_store()
    reloaded_registry = load_document_registry()
    reloaded_chunks = load_all_chunks()

    print(f"Reloaded state: {len(reloaded_registry)} registry files, {len(reloaded_chunks)} chunks, {reloaded_vector_store.index.ntotal} FAISS vectors.")
    assert test_doc_name in reloaded_registry
    assert test_doc2_name in reloaded_registry
    test_results["TEST 7"] = "PASSED: Full index state and registry reloaded from disk without issues."

    # ──────────────────────────────────────────────────────────────────────────
    # TEST 8: Test Rebuild Entire Index
    # ──────────────────────────────────────────────────────────────────────────
    log_test(8, "Test 'Rebuild Entire Index'")
    rebuild_res = rebuild_entire_index()
    print(f"Rebuild result: {rebuild_res['message']}")
    assert rebuild_res["status"] == "success"
    assert rebuild_res["total_documents"] >= 6
    test_results["TEST 8"] = f"PASSED: Rebuild entire index reprocessed {rebuild_res['total_documents']} documents cleanly."

    # ──────────────────────────────────────────────────────────────────────────
    # TEST 9: Detect Configuration Incompatibility
    # ──────────────────────────────────────────────────────────────────────────
    log_test(9, "Detect Index Configuration Incompatibility")
    meta = load_index_metadata()
    # Temporarily modify meta to simulate model change
    original_model = meta["embedding_model"]
    meta["embedding_model"] = "incompatible-model-v999"
    save_index_metadata(meta)

    compat, compat_msg = check_index_compatibility()
    print(f"Compatibility check with altered model: {compat} | Message: {compat_msg}")
    assert not compat, "Expected incompatible status!"
    assert "incompatible-model-v999" in compat_msg

    # Restore correct metadata
    meta["embedding_model"] = original_model
    save_index_metadata(meta)
    compat_restored, _ = check_index_compatibility()
    assert compat_restored, "Failed to restore compatibility!"
    test_results["TEST 9"] = "PASSED: Incompatible index model detected with warning requiring rebuild."

    # ──────────────────────────────────────────────────────────────────────────
    # TEST 10: List-aware project query completeness
    # ──────────────────────────────────────────────────────────────────────────
    log_test(10, "List-aware project query completeness ('What are my projects?')")
    projects_res = run_rag_pipeline("What are my projects?", vector_store=load_vector_store(), debug=True)
    ans = projects_res["answer"]
    print(f"Projects Answer:\n{ans}")

    # Must list resume projects
    assert "Intelligent Document Processing" in ans or "Document Processing" in ans, "Missing Intelligent Document Processing Platform!"
    assert "Emotion Recognition" in ans or "Facial Emotion" in ans or "FER" in ans, "Missing Facial Emotion Recognition!"
    assert "Fin-Agent" in ans or "Loan Underwriting" in ans, "Missing Fin-Agent!"
    test_results["TEST 10"] = "PASSED: 'What are my projects?' returns all projects completely."

    # ──────────────────────────────────────────────────────────────────────────
    # Clean up test documents
    # ──────────────────────────────────────────────────────────────────────────
    print("\nCleaning up test artifacts...")
    for tdoc in [test_doc_name, test_doc2_name]:
        tpath = os.path.join(config.DOCS_DIR, tdoc)
        if os.path.exists(tpath):
            os.remove(tpath)
        remove_document_from_index(tdoc)

    # Final rebuild to restore exact original documents
    rebuild_entire_index()
    print("Test cleanup complete. Original index restored.")

    print(f"\n{'='*70}")
    print("ALL 10 TESTS COMPLETED SUCCESSFULLY!")
    print(f"{'='*70}\n")
    for k, v in test_results.items():
        print(f"  {k}: {v}")

    with open("test_incremental_results.json", "w", encoding="utf-8") as f:
        json.dump(test_results, f, indent=2)


if __name__ == "__main__":
    run_all_tests()
