"""
Auto-generate an eval set from your ingested chunks using the LLM itself.

Why: hand-writing 100-200 question/answer pairs isn't realistic. Instead,
for each chunk, we ask the LLM to write 1-3 questions that chunk answers,
plus a short expected-keyword phrase pulled directly from that chunk's
text. This gives you a large, source-grounded eval set fast — you then
spot-check a sample rather than validating all of them by hand.

Caveat: LLM-generated questions are usually "easier" than the adversarial
ones you write by hand (they tend to closely mirror the chunk's own
wording), so treat this as a breadth/coverage check, not a replacement
for the harder manual cases already in eval.py.

Run:
    python generate_eval_questions.py --source "bloch" --per-chunk 2
"""

import os
import json
import argparse
import time
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate

load_dotenv()

from query import load_chunks
import config

GEN_PROMPT = """You are creating a test set to evaluate a document
retrieval system. Given the text chunk below, write {n} question(s) that
this chunk directly and unambiguously answers.

Rules:
- Phrase each question the way a real reader would naturally ask it — as
  if asking about the paper/document's subject matter. NEVER reference
  "the chunk", "the text", "this passage", or similar meta-language in
  the question itself.
- Each question must be answerable using ONLY this chunk's content, and
  the fact must be EXPLICITLY present in the chunk — never invent a
  question about a field that isn't actually there (e.g. don't ask for
  an ISBN if only a DOI is present).
- Provide a short "expected_keyword" for each question: a distinctive
  word or phrase (2-6 words) copied EXACTLY from the chunk that would
  appear in a correct answer. Prefer specific terms, numbers, or names
  over generic words.
- Skip entirely (respond with []) if the chunk is boilerplate: copyright
  notices, "authorized licensed use" footers, page numbers, download
  timestamps, or a references/citation list with no substantive claim.
- Respond ONLY with valid JSON, no other text, in this exact format:
  [{{"question": "...", "expected_keyword": "..."}}, ...]
  If no good question exists for this chunk, respond with: []

Chunk:
{chunk_text}
"""

BOILERPLATE_MARKERS = [
    "authorized licensed use limited",
    "downloaded on",
    "restrictions apply",
]


def is_boilerplate(chunk_text):
    """Quick heuristic to skip repeated footer/copyright noise before even
    calling the LLM — saves API calls and avoids hallucinated questions
    about fields (like ISBN) that only appear in junk boilerplate text."""
    text_lower = chunk_text.lower()
    return any(marker in text_lower for marker in BOILERPLATE_MARKERS)


def generate_questions_for_chunk(llm, chunk_text, n=2):
    prompt = ChatPromptTemplate.from_template(GEN_PROMPT)
    chain = prompt | llm
    try:
        response = chain.invoke({"chunk_text": chunk_text, "n": n})
        text = response.content.strip()
        # Strip markdown code fences if the model added them anyway
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        return json.loads(text)
    except (json.JSONDecodeError, IndexError) as e:
        print(f"  ! Skipped a chunk (couldn't parse LLM output): {e}")
        return []


def main(source_filter, per_chunk, max_chunks, output_path):
    chunks = load_chunks()

    if source_filter:
        chunks = [
            c for c in chunks
            if source_filter.lower() in c.metadata.get("source", "").lower()
        ]

    if max_chunks:
        chunks = chunks[:max_chunks]

    print(f"Generating questions from {len(chunks)} chunks...")

    api_key = os.getenv("GROQ_API_KEY")
    llm = ChatGroq(model="mixtral-8x7b-32768", temperature=0.3, api_key=api_key)

    all_questions = []
    skipped_boilerplate = 0
    for i, chunk in enumerate(chunks, 1):
        source = chunk.metadata.get("source", "unknown")

        if is_boilerplate(chunk.page_content):
            skipped_boilerplate += 1
            print(f"  [{i}/{len(chunks)}] {source} -> skipped (boilerplate)")
            continue

        qa_pairs = generate_questions_for_chunk(llm, chunk.page_content, n=per_chunk)

        for qa in qa_pairs:
            all_questions.append({
                "question": qa.get("question", ""),
                "expected_source": source,
                "expected_keywords": [qa.get("expected_keyword", "")],
            })

        print(f"  [{i}/{len(chunks)}] {source} -> {len(qa_pairs)} question(s)")
        time.sleep(0.3)  # be gentle on rate limits

    print(f"\n(Skipped {skipped_boilerplate} boilerplate chunks without calling the LLM)")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_questions, f, indent=2)

    print(f"\n✓ Generated {len(all_questions)} questions -> saved to '{output_path}'")
    print(f"  Run 'python eval.py --set {output_path}' to test against them.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=None,
                         help="Only generate from chunks whose source contains this string")
    parser.add_argument("--per-chunk", type=int, default=2,
                         help="Questions to attempt per chunk (default 2)")
    parser.add_argument("--max-chunks", type=int, default=None,
                         help="Cap number of chunks processed (for a quick test run)")
    parser.add_argument("--output", default="eval_questions.json")
    args = parser.parse_args()

    main(args.source, args.per_chunk, args.max_chunks, args.output)