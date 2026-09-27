"""
Phase 4 — Document Type Classification.

Classifies documents dynamically based on content (not just filename):
- resume
- academic
- research_paper
- project
- certificate
- other
"""

from __future__ import annotations

import re
import os
from typing import Dict, Any


def classify_document_content(text: str, filename: str = "") -> str:
    """
    Classify document type using text patterns and content heuristics.
    """
    t = text.lower()
    fn = filename.lower()

    # Academic transcript / Grade card signals
    academic_signals = [
        r"\bsgpa\b", r"\bmarks card\b", r"\bgrade card\b", r"\bsemester grade point\b",
        r"\bregistration no\b", r"\bcourse assessment\b", r"\bend semester\b",
        r"\bexamination\b", r"\btotal credits\b", r"\bsemester\s*:\s*(?:i|ii|iii|iv|v|vi|vii|viii|\d+)\b"
    ]
    academic_score = sum(1 for p in academic_signals if re.search(p, t))

    # Resume signals
    resume_signals = [
        r"\beducation\b", r"\bexperience\b", r"\bprojects\b", r"\btechnical skills\b",
        r"\bcertifications\b", r"\bachievements\b", r"\bgithub\b", r"\blinkedin\b",
        r"\bleetcode\b", r"\bcurriculum vitae\b", r"\bresume\b"
    ]
    resume_score = sum(1 for p in resume_signals if re.search(p, t))

    # Research paper signals
    research_signals = [
        r"\babstract\b", r"\bieee\b", r"\bacm\b", r"\bproceedings\b", r"\bconference\b",
        r"\bdoi\b", r"\barxiv\b", r"\bindex terms\b", r"\breferences\b",
        r"\bconclusion\b", r"\bet al\b", r"\bfig\.\s*\d+\b"
    ]
    research_score = sum(1 for p in research_signals if re.search(p, t))

    # Certificate / Online Course signals
    certificate_signals = [
        r"\bcertificate\b", r"\bcerti[\s\x00]*cate\b", r"\bthis is to certify\b",
        r"\bhas successfully completed\b", r"\bcredential\b", r"\bawarded to\b",
        r"\bcoursera\b", r"\budemy\b", r"\bedx\b", r"\bdeeplearning\.ai\b",
        r"\bonline course\b", r"\bauthorized by\b", r"\bverify at\b",
        r"\bcompleted an online course\b", r"\bparticipation in the course\b"
    ]
    certificate_score = sum(1 for p in certificate_signals if re.search(p, t))

    # Evaluation
    # If transcript has clear academic signals (like marks card, SGPA, course assessment):
    if academic_score >= 3 or ("registration no" in t and "sgpa" in t):
        return "academic"

    # If it has strong resume structural sections
    if resume_score >= 4 or ("curriculum vitae" in t or "resume" in fn):
        # Double check it's not a marks card that happened to have some words
        if academic_score >= 4 and "course assessment" in t:
            return "academic"
        return "resume"

    # Certificate or Course Completion (check before research paper, especially for short/cert docs)
    if (
        certificate_score >= 2 or
        any(k in fn for k in ["cert", "coursera", "udemy", "credential", "course"]) or
        ("coursera" in t and ("completed" in t or "course" in t)) or
        ("successfully completed" in t)
    ):
        return "certificate"

    # Research paper
    if research_score >= 3 or ("abstract" in t and "references" in t):
        return "research_paper"

    # Specific project writeup
    if "project overview" in t or "architecture diagram" in t:
        return "project"

    return "other"

