from __future__ import annotations
import os, sys, json, warnings, math
from typing import List, Dict, Any, Optional
from datetime import datetime
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"): sys.stderr.reconfigure(encoding="utf-8")
from dotenv import load_dotenv
load_dotenv()

RAGAS_METRICS_DESCRIPTIONS = {
    "faithfulness": "Is the answer grounded in retrieved context? 1.0=fully supported.",
    "answer_relevancy": "Does answer address the question? 1.0=perfectly relevant.",
    "context_precision": "Are retrieved chunks useful? High=low retrieval noise.",
    "context_recall": "Does context cover the full answer? Needs reference answer.",
}

def setup_langsmith_tracing():
    api_key = os.getenv("LANGSMITH_API_KEY")
    project  = os.getenv("LANGSMITH_PROJECT", "RAG-Assistant-Eval")
    endpoint = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
    if not api_key:
        print("[LangSmith] No LANGSMITH_API_KEY found -- tracing disabled.")
        return False
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"]    = api_key
    os.environ["LANGCHAIN_PROJECT"]    = project
    os.environ["LANGCHAIN_ENDPOINT"]   = endpoint
    print(f"[LangSmith] Tracing enabled, project={project!r}")
    return True

def disable_langsmith_tracing():
    os.environ["LANGCHAIN_TRACING_V2"] = "false"

def _get_ragas_llm():
    from langchain_groq import ChatGroq
    import config
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not set.")
    return ChatGroq(model=getattr(config, "GROQ_MODEL", "openai/gpt-oss-120b"),
                    temperature=0, api_key=api_key, max_retries=2)

def _get_ragas_embeddings():
    from langchain_community.embeddings import HuggingFaceEmbeddings
    import config
    return HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)

def run_ragas_evaluation(questions, reference_answers=None, vector_store=None,
                          metrics_to_run=None, progress_callback=None):
    def _prog(msg, p=0.0):
        if progress_callback:
            progress_callback(msg, p)
        else:
            pct = int(p * 100)
            print(f"[RAGAS] ({pct}%) {msg}")
    errors = []
    has_refs = bool(reference_answers and len(reference_answers) == len(questions))
    if metrics_to_run is None:
        metrics_to_run = (
            ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
            if has_refs else
            ["faithfulness", "answer_relevancy", "context_precision"]
        )
    _prog("Loading RAGAS metrics and LLM wrappers...", 0.05)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from ragas.metrics.collections import (
            Faithfulness, AnswerRelevancy, LLMContextPrecisionWithoutReference)
        try:
            from ragas.metrics.collections import LLMContextRecall
            recall_ok = True
        except ImportError:
            recall_ok = False
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    llm_w   = LangchainLLMWrapper(_get_ragas_llm())
    embed_w = LangchainEmbeddingsWrapper(_get_ragas_embeddings())
    metric_objects = []
    if "faithfulness" in metrics_to_run:
        metric_objects.append(Faithfulness(llm=llm_w))
    if "answer_relevancy" in metrics_to_run:
        metric_objects.append(AnswerRelevancy(llm=llm_w, embeddings=embed_w))
    if "context_precision" in metrics_to_run:
        metric_objects.append(LLMContextPrecisionWithoutReference(llm=llm_w))
    if "context_recall" in metrics_to_run and recall_ok and has_refs:
        metric_objects.append(LLMContextRecall(llm=llm_w))
    _prog("Running RAG pipeline on each question...", 0.10)
    from query import load_vector_store, run_rag_pipeline
    if vector_store is None:
        vector_store = load_vector_store()
    samples_data = []
    n = len(questions)
    for i, question in enumerate(questions):
        _prog(f"Q{i+1}/{n}: {question[:60]}...", 0.10 + 0.50 * (i / n))
        try:
            result = run_rag_pipeline(question=question, vector_store=vector_store)
            answer = result["answer"]
            contexts = []
            for pkg in result.get("sub_query_packages", []):
                for chunk in pkg.get("chunks", []):
                    doc = chunk.get("doc")
                    if doc:
                        contexts.append(doc.page_content.strip())
            sample = {"question": question, "answer": answer, "contexts": contexts}
            if has_refs:
                sample["reference"] = reference_answers[i]
            samples_data.append(sample)
        except Exception as e:
            errors.append({"question": question, "error": str(e)})
            _prog(f"  Warning: Q{i+1} failed: {e}", 0.10 + 0.50 * (i / n))
    if not samples_data:
        return {"scores": {}, "per_question": [], "samples": [],
                "timestamp": datetime.now().isoformat(), "errors": errors,
                "error": "All questions failed."}
    _prog("Building RAGAS EvaluationDataset...", 0.65)
    from ragas import EvaluationDataset, SingleTurnSample
    ragas_samples = []
    for s in samples_data:
        kw = {"user_input": s["question"], "response": s["answer"],
              "retrieved_contexts": s["contexts"]}
        if "reference" in s:
            kw["reference"] = s["reference"]
        ragas_samples.append(SingleTurnSample(**kw))
    dataset = EvaluationDataset(samples=ragas_samples)
    _prog("Running RAGAS scoring, this may take 30-60s...", 0.70)
    try:
        from ragas import evaluate, RunConfig
        ragas_result = evaluate(dataset=dataset, metrics=metric_objects,
                                run_config=RunConfig(timeout=120, max_retries=2, max_wait=30))
        df = ragas_result.to_pandas()
        metric_names = [m.name for m in metric_objects]
        aggregated = {}
        for mn in metric_names:
            if mn in df.columns:
                col = df[mn].dropna()
                aggregated[mn] = round(float(col.mean()), 4) if len(col) > 0 else None
        per_q = []
        for idx, row in df.iterrows():
            entry = {"question": samples_data[idx]["question"], "answer": samples_data[idx]["answer"]}
            for mn in metric_names:
                if mn in row:
                    val = row[mn]
                    try:
                        entry[mn] = round(float(val), 4) if val is not None and not math.isnan(float(val)) else None
                    except Exception:
                        entry[mn] = None
            per_q.append(entry)
    except Exception as e:
        errors.append({"step": "ragas_evaluate", "error": str(e)})
        _prog(f"RAGAS evaluate failed: {e}", 1.0)
        return {"scores": {}, "per_question": [], "samples": samples_data,
                "timestamp": datetime.now().isoformat(), "errors": errors,
                "error": f"RAGAS evaluate() failed: {e}"}
    _prog("Logging results to LangSmith...", 0.95)
    _log_to_langsmith(aggregated, per_q, questions)
    _prog("RAGAS evaluation complete!", 1.0)
    return {"scores": aggregated, "per_question": per_q, "samples": samples_data,
            "timestamp": datetime.now().isoformat(), "errors": errors,
            "metrics_run": [m.name for m in metric_objects],
            "descriptions": {k: v for k, v in RAGAS_METRICS_DESCRIPTIONS.items() if k in aggregated}}

def _log_to_langsmith(aggregated, per_q, questions):
    api_key = os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")
    if not api_key:
        return
    try:
        from langsmith import Client
        client  = Client(api_key=api_key)
        project = os.getenv("LANGSMITH_PROJECT", "RAG-Assistant-Eval")
        run_id  = client.create_run(
            name="RAGAS-Evaluation", run_type="chain",
            inputs={"questions": questions, "num_questions": len(questions)},
            outputs={"aggregated_scores": aggregated, "per_question": per_q},
            extra={"metadata": {"ragas_version": "0.4.x", "timestamp": datetime.now().isoformat()}},
            project_name=project)
        client.update_run(run_id, end_time=datetime.now())
        print(f"[LangSmith] Logged eval run, project={project!r}")
    except Exception as e:
        print(f"[LangSmith] Logging failed (non-critical): {e}")

DEFAULT_TEST_QUESTIONS = [
    "What are my coursework subjects?",
    "How well am I doing academically overall?",
    "Where do I currently work?",
    "What model architecture did I use for emotion detection?",
    "Where did I study before university?",
]

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run RAGAS + LangSmith evaluation")
    parser.add_argument("--questions", default=None, help="Path to JSON eval set")
    parser.add_argument("--metrics", nargs="+",
                        default=["faithfulness", "answer_relevancy", "context_precision"])
    parser.add_argument("--langsmith", action="store_true", help="Enable LangSmith tracing")
    parser.add_argument("--output", default="ragas_results.json")
    args = parser.parse_args()
    if args.langsmith:
        setup_langsmith_tracing()
    questions = DEFAULT_TEST_QUESTIONS
    if args.questions:
        with open(args.questions, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        if loaded and isinstance(loaded[0], dict):
            questions = [q.get("question", "") for q in loaded if q.get("question")]
        else:
            questions = loaded
    print(f"Running RAGAS evaluation on {len(questions)} questions...")
    results = run_ragas_evaluation(questions=questions, metrics_to_run=args.metrics)
    print("=" * 60)
    print("RAGAS EVALUATION RESULTS")
    print("=" * 60)
    for metric, score in results["scores"].items():
        bar = "X" * int((score or 0) * 20)
        print(f"  {metric:<28} {score:.4f}  [{bar:<20}]")
    print("=" * 60)
    if results.get("errors"):
        print(f"Errors ({len(results['errors'])}):")
        for e in results["errors"]:
            print(f"  - {e}")
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Full results saved to {args.output!r}")