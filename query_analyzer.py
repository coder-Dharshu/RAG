"""
Phase 6 & 7 — Query Understanding and Query Decomposition.

Analyzes user queries to determine:
- Intent
- Question Type
- Entity / Section
- Filter Criteria
- Source Hints
- Compound Question Decomposition
"""

from __future__ import annotations

import os
import re
import json
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
load_dotenv()
import config


ANALYZER_SYSTEM_PROMPT = """You are an expert Query Understanding and Decomposition engine for a Personal RAG system.
The system contains personal documents:
1. Resume (education, coursework, experience, projects, skills, certifications, achievements)
2. Academic transcripts / grade cards (SGPA, CGPA, courses, grades, registration number)
3. Certificates & online courses (Coursera, Udemy, certifications, training, completion credentials)
4. Research papers (e.g. quantum computing, Bloch sphere, FPGA emulation, methodology, tools)
5. Other documents (company policy, general notes, newly uploaded files)

Analyze the user's query and decompose compound questions into independent sub-queries.

Output ONLY a JSON array of objects (no markdown, no preamble) where each object has:
- "query": focused, self-contained sub-question
- "intent": one of ["personal_academic", "personal_projects", "personal_experience", "personal_skills", "research", "document_summary", "comparison", "general_document_query"]
- "question_type": one of ["factual", "list", "filtered_list", "explanation", "summary", "comparison", "multi_hop"]
- "source_hint": array of strings from ["resume", "academic", "certificate", "research_paper", "other"]
- "section": target section if known, e.g. "projects", "education", "coursework", "certifications", "skills", "experience", "course_grades", "methodology", or null
- "filter_criteria": filter keyword if question_type is "filtered_list" (e.g. "Python", "AI"), or null
- "entity": specific named entity if mentioned (e.g. "Intelligent Document Processing Platform"), or null

CRITICAL RULES:
1. If the question asks for ALL projects or what projects the user has done, question_type MUST be "list" and section MUST be "projects".
2. If asking for projects matching a specific technology/domain (e.g. "Which of my projects use Python?", "projects involving AI"), question_type MUST be "filtered_list", section MUST be "projects", and filter_criteria MUST be set (e.g. "Python", "AI").
3. If asking for courses, coursework, certifications, credentials, or training (e.g. Coursera, Udemy, online courses), source_hint MUST include ["resume", "certificate", "other"] so that both resume listings and uploaded certificate documents are searched.
3b. If asking for subjects, courses, marks, or grades for a specific semester or academic term (e.g. "subjects in sem 5", "what subjects did I take in sem5", "courses in semester 6"), section MUST be "course_grades", source_hint MUST include ["academic"], question_type MUST be "factual" or "list", and filter_criteria MUST be null.
4. If the user asks about an uploaded document or mentions words matching an uploaded file (e.g. "genai coursera", "from the uploaded file", "not from resume"), source_hint MUST include ["certificate", "other", "research_paper"] to search newly uploaded files.
5. For personal questions (my CGPA, my projects, my experience, my skills), source_hint should target ["resume", "academic", "certificate", "other"], but NOT "research_paper" unless explicitly asking about a paper.
6. Split compound questions cleanly so that each sub-query focuses strictly on ONE topic.
7. If the question is simple/single-intent, return an array of 1 object.
"""

ANALYZER_FEW_SHOT = """User: "What is my CGPA and what are my projects and what tools were used in the quantum Bloch sphere paper?"
Output:
[
  {
    "query": "What is my CGPA?",
    "intent": "personal_academic",
    "question_type": "factual",
    "source_hint": ["resume", "academic"],
    "section": "education",
    "filter_criteria": null,
    "entity": "CGPA"
  },
  {
    "query": "What are my projects?",
    "intent": "personal_projects",
    "question_type": "list",
    "source_hint": ["resume"],
    "section": "projects",
    "filter_criteria": null,
    "entity": null
  },
  {
    "query": "What tools and hardware/software components were used in the quantum Bloch sphere research paper?",
    "intent": "research",
    "question_type": "factual",
    "source_hint": ["research_paper"],
    "section": "methodology",
    "filter_criteria": null,
    "entity": "quantum Bloch sphere"
  }
]"""


def _classify_single_query_rule_based(q: str) -> Dict[str, Any]:
    """Deterministic rule-based query classifier fallback."""
    text = q.lower()

    # Detect filtered list (e.g. projects with Python / AI)
    proj_filter_match = re.search(r"projects?\s+(?:that\s+use|using|with|involving|in)\s+([a-zA-Z\+\#]+)", text)
    which_proj_filter = re.search(r"which\s+(?:of\s+my\s+)?projects?\s+(?:use|involve|have)\s+([a-zA-Z\+\#]+)", text)

    filter_kw = None
    if proj_filter_match:
        filter_kw = proj_filter_match.group(1).strip()
    elif which_proj_filter:
        filter_kw = which_proj_filter.group(1).strip()
    elif "project" in text and "python" in text:
        filter_kw = "Python"
    elif "project" in text and "ai" in text:
        filter_kw = "AI"

    if filter_kw:
        return {
            "query": q,
            "intent": "personal_projects",
            "question_type": "filtered_list",
            "source_hint": ["resume"],
            "section": "projects",
            "filter_criteria": filter_kw,
            "entity": None,
        }

    # Projects list questions
    if any(p in text for p in ["what are my projects", "list my projects", "list all my projects", "all my projects", "which projects have i done", "what projects have i done"]):
        return {
            "query": q,
            "intent": "personal_projects",
            "question_type": "list",
            "source_hint": ["resume"],
            "section": "projects",
            "filter_criteria": None,
            "entity": None,
        }

    # Technologies in projects
    if "technologies" in text and "project" in text:
        return {
            "query": q,
            "intent": "personal_projects",
            "question_type": "list",
            "source_hint": ["resume"],
            "section": "projects",
            "filter_criteria": None,
            "entity": "technologies",
        }

    # Specific project inquiry
    if ("tell me about my" in text or "describe my" in text) and ("platform" in text or "system" in text or "agent" in text):
        return {
            "query": q,
            "intent": "personal_projects",
            "question_type": "explanation",
            "source_hint": ["resume"],
            "section": "projects",
            "filter_criteria": None,
            "entity": q.replace("Tell me about my", "").replace("Describe my", "").strip("?. "),
        }

    # Coursework & Online Courses / Certifications (Coursera, Udemy, etc.)
    if any(w in text for w in ["coursera", "udemy", "edx", "certif", "credential", "online course"]) or ("course" in text and "genai" in text):
        return {
            "query": q,
            "intent": "personal_academic",
            "question_type": "factual",
            "source_hint": ["resume", "certificate", "other"],
            "section": "certifications",
            "filter_criteria": None,
            "entity": "course",
        }

    # Explicit mention of uploaded file or not from resume
    if any(w in text for w in ["uploaded file", "uploaded doc", "not from resume", "from the file", "from the document"]):
        return {
            "query": q,
            "intent": "general_document_query",
            "question_type": "factual",
            "source_hint": ["certificate", "other", "research_paper", "resume"],
            "section": None,
            "filter_criteria": None,
            "entity": None,
        }

    # Coursework (specifically matches resume COURSEWORK section)
    if "coursework" in text:
        return {
            "query": q,
            "intent": "personal_academic",
            "question_type": "list",
            "source_hint": ["resume", "certificate"],
            "section": "coursework",
            "filter_criteria": None,
            "entity": "coursework",
        }

    # Semester subjects / courses (transcripts)
    if (any(w in text for w in ["subject", "subjects", "course", "courses", "paper", "papers"]) and any(s in text for s in ["sem", "semester"])) or \
       ("sem" in text and any(w in text for w in ["have", "take", "took", "studied", "subjects", "courses"])):
        return {
            "query": q,
            "intent": "personal_academic",
            "question_type": "factual",
            "source_hint": ["academic"],
            "section": "course_grades",
            "filter_criteria": None,
            "entity": "semester subjects",
        }

    # CGPA / SGPA / Academic
    if any(w in text for w in ["cgpa", "sgpa", "sgps", "gpa", "grade", "marks", "subjects", "degree"]):
        return {
            "query": q,
            "intent": "personal_academic",
            "question_type": "factual",
            "source_hint": ["academic", "resume"],
            "section": "academic_performance" if any(w in text for w in ["cgpa", "sgpa", "sgps", "gpa"]) else "course_grades",
            "filter_criteria": None,
            "entity": "CGPA" if any(w in text for w in ["cgpa", "sgpa", "sgps"]) else "grades",
        }

    # Research paper queries (Bloch sphere, quantum, FPGA, OpenCL)
    if any(w in text for w in ["bloch", "quantum", "fpga", "research paper", "paper", "opencl", "simulation", "emulation"]):
        return {
            "query": q,
            "intent": "research",
            "question_type": "factual",
            "source_hint": ["research_paper"],
            "section": "methodology",
            "filter_criteria": None,
            "entity": "Bloch sphere quantum paper",
        }

    # Experience / Internships
    if any(w in text for w in ["experience", "work", "job", "intern", "internship", "company"]):
        return {
            "query": q,
            "intent": "personal_experience",
            "question_type": "list" if "list" in text else "factual",
            "source_hint": ["resume"],
            "section": "experience",
            "filter_criteria": None,
            "entity": None,
        }

    # Skills
    if any(w in text for w in ["skill", "skills", "languages", "tools", "frameworks"]):
        return {
            "query": q,
            "intent": "personal_skills",
            "question_type": "list",
            "source_hint": ["resume"],
            "section": "skills",
            "filter_criteria": None,
            "entity": None,
        }

    return {
        "query": q,
        "intent": "general_document_query",
        "question_type": "factual",
        "source_hint": ["resume", "academic", "research_paper", "other"],
        "section": None,
        "filter_criteria": None,
        "entity": None,
    }


def _split_compound_rule_based(question: str) -> List[str]:
    """Split compound questions on conjunctions and punctuation."""
    # Split on patterns like: "X and what is Y", "X, what are Y", "X? What is Y"
    parts = re.split(
        r"(?:,\s*(?:and\s+)?|\s+and\s+)(?=(?:what|which|how|where|tell|list|describe|who)\b)",
        question,
        flags=re.IGNORECASE,
    )
    if len(parts) <= 1:
        parts = re.split(r"[;\n]+", question)

    clean_parts = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if not p.endswith("?"):
            p += "?"
        p = p[0].upper() + p[1:]
        clean_parts.append(p)

    return clean_parts if clean_parts else [question]


def decompose_and_analyze(question: str, use_llm: bool = True) -> List[Dict[str, Any]]:
    """
    Decompose question into sub-queries with rich query understanding metadata.
    """
    if use_llm:
        api_key = os.getenv("GROQ_API_KEY")
        if api_key:
            try:
                from langchain_groq import ChatGroq
                from langchain_core.messages import SystemMessage, HumanMessage

                models_to_try = [
                    getattr(config, "GROQ_MODEL", "openai/gpt-oss-120b"),
                    getattr(config, "GROQ_FALLBACK_MODEL", "openai/gpt-oss-20b"),
                ]

                messages = [
                    SystemMessage(content=ANALYZER_SYSTEM_PROMPT),
                    HumanMessage(content=f"User: {ANALYZER_FEW_SHOT.split('Output:')[0].replace('User: ', '').strip()}"),
                    HumanMessage(content=f"Output:\n{ANALYZER_FEW_SHOT.split('Output:')[1].strip()}"),
                    HumanMessage(content=f"User: \"{question}\""),
                ]

                parsed = None
                for m in models_to_try:
                    try:
                        llm = ChatGroq(
                            model=m,
                            temperature=0,
                            api_key=api_key,
                            max_tokens=600,
                            max_retries=2,
                        )
                        response = llm.invoke(messages)
                        content = response.content.strip()

                        # Clean markdown fences if any
                        content = re.sub(r"^```[a-z]*\n?", "", content)
                        content = re.sub(r"\n?```$", "", content)

                        parsed = json.loads(content)
                        if isinstance(parsed, list) and parsed:
                            break
                    except Exception as err:
                        print(f"[query_analyzer] Model {m} failed: {err}")
                        continue
                if isinstance(parsed, list) and parsed:
                    # Sanity check items
                    for item in parsed:
                        item.setdefault("source_hint", ["other"])
                        if isinstance(item["source_hint"], str):
                            item["source_hint"] = [item["source_hint"]]
                        item.setdefault("section", None)
                        item.setdefault("filter_criteria", None)
                        item.setdefault("entity", None)
                        item.setdefault("question_type", "factual")
                        item.setdefault("intent", "general_document_query")
                    return parsed
            except Exception as e:
                print(f"[query_analyzer] LLM analysis failed: {e} — using rule-based fallback")

    # Fallback
    sub_questions = _split_compound_rule_based(question)
    analyzed = [_classify_single_query_rule_based(sq) for sq in sub_questions]
    return analyzed
