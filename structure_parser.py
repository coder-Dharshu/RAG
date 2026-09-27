"""
Phase 2, 3, 4, 5 — Structure-Aware Parsing, Chunking, Metadata Enrichment, and Section Indexing.

Implements:
- parse_document(file_path)
- detect_document_structure(file_path, content, doc_type)
- extract_sections(text, doc_type)
- structure_aware_chunking()
- enrich_metadata()
- build_section_index(chunks)
"""

from __future__ import annotations

import os
import re
import sys
import json
from typing import List, Dict, Any, Tuple, Optional
from langchain_core.documents import Document

from document_classifier import classify_document_content


# ── Heading and section regexes for Resumes ────────────────────────────────────
RESUME_SECTION_HEADINGS = {
    "EDUCATION": "education",
    "ACADEMIC BACKGROUND": "education",
    "COURSEWORK": "coursework",
    "RELEVANT COURSEWORK": "coursework",
    "PROFESSIONAL EXPERIENCE": "experience",
    "WORK EXPERIENCE": "experience",
    "EXPERIENCE": "experience",
    "INTERNSHIPS": "experience",
    "PROJECTS": "projects",
    "ACADEMIC PROJECTS": "projects",
    "KEY PROJECTS": "projects",
    "PERSONAL PROJECTS": "projects",
    "TECHNICAL SKILLS": "skills",
    "SKILLS": "skills",
    "SKILLS & TOOLS": "skills",
    "CERTIFICATIONS": "certifications",
    "CERTIFICATES": "certifications",
    "ACHIEVEMENTS": "achievements",
    "HONORS & AWARDS": "achievements",
    "AWARDS": "achievements",
    "PUBLICATIONS": "publications",
    "LEADERSHIP": "leadership",
}

# Regex to detect project title lines in a resume:
# e.g., "Intelligent Document Processing Platform | Python, FastAPI, Redis, Celery, Docker, OpenCV, Qwen 3 30B Github May 2026"
# e.g., "Real-Time Facial Emotion Recognition (FER) System | Python, PyTorch/TensorFlow, OpenCV... July 2026"
# e.g., "Fin-Agent – Multi-Stage Loan Underwriting System | React, Vite... April 2026"
# Note: handles |, –, —, and - separators, as well as GitHub/GitLab links and dates.
PROJECT_LINE_RE = re.compile(
    r"^([A-Za-z0-9][^\n]{3,80}?)\s*(?:\||–|—|-)\s*(.+)$"
)

# Academic transcript key-value pattern
KV_LINE_RE = re.compile(r"^([A-Za-z0-9\s/&.\-\(\)]+?)\s*:\s*(.+)$")


def _clean_stem(path: str) -> str:
    base = os.path.basename(path).lower()
    return re.sub(r"[^a-z0-9]+", "_", os.path.splitext(base)[0]).strip("_")


# ── 1. Resume Parser ──────────────────────────────────────────────────────────
def parse_resume(pdf_path: str) -> List[Document]:
    """
    Parse a resume PDF into structure-aware, metadata-rich Documents.
    Preserves complete project chunks (Title + Tech + Description + Bullets).
    """
    import pdfplumber

    pages_text: List[Tuple[int, str]] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):
            text = page.extract_text() or ""
            pages_text.append((page_num, text))

    all_lines: List[Tuple[str, int]] = []
    for page_num, text in pages_text:
        for line in text.split("\n"):
            all_lines.append((line, page_num))

    stem = _clean_stem(pdf_path)
    documents: List[Document] = []

    current_section = "contact_info"
    current_section_lines: List[str] = []
    current_project_name: Optional[str] = None
    current_project_lines: List[str] = []
    current_page = 1
    proj_counter = 0
    sec_counter = 0

    def flush_project():
        nonlocal current_project_name, current_project_lines, proj_counter
        if current_project_name and current_project_lines:
            proj_counter += 1
            content = current_project_name + "\n" + "\n".join(current_project_lines)
            
            # Clean title: keep entire project name before tech separator '|'
            clean_title = current_project_name.split("|")[0].strip()
            # Normalize dashes
            clean_title = clean_title.replace("\u2013", "–").replace("\u2014", "—")

            chunk_id = f"{stem}_projects_{proj_counter:03d}"
            documents.append(Document(
                page_content=content.strip(),
                metadata={
                    "source": pdf_path,
                    "filename": os.path.basename(pdf_path),
                    "page": current_page,
                    "document_type": "resume",
                    "section": "projects",
                    "subsection": None,
                    "item_type": "project",
                    "item_name": clean_title,
                    "chunk_id": chunk_id,
                }
            ))
        current_project_name = None
        current_project_lines = []

    def flush_section():
        nonlocal current_section_lines, sec_counter
        if current_section_lines and current_section != "projects":
            sec_counter += 1
            content = "\n".join(current_section_lines).strip()
            if content:
                chunk_id = f"{stem}_{current_section}_{sec_counter:03d}"
                documents.append(Document(
                    page_content=content,
                    metadata={
                        "source": pdf_path,
                        "filename": os.path.basename(pdf_path),
                        "page": current_page,
                        "document_type": "resume",
                        "section": current_section,
                        "subsection": None,
                        "item_type": "section",
                        "item_name": current_section.replace("_", " ").title(),
                        "chunk_id": chunk_id,
                    }
                ))
        current_section_lines = []

    in_projects = False

    for raw_line, page_num in all_lines:
        current_page = page_num
        line = raw_line.strip()
        if not line:
            continue

        # Check if line is a known resume section heading
        upper = line.upper()
        matched_section = None
        if upper in RESUME_SECTION_HEADINGS:
            matched_section = RESUME_SECTION_HEADINGS[upper]
        elif len(line) <= 45 and line.isupper() and any(k in upper for k in RESUME_SECTION_HEADINGS):
            for k, v in RESUME_SECTION_HEADINGS.items():
                if k in upper:
                    matched_section = v
                    break

        if matched_section:
            if in_projects:
                flush_project()
                in_projects = False
            else:
                flush_section()

            current_section = matched_section
            in_projects = (current_section == "projects")
            continue

        if in_projects:
            # Check if this line starts a new project
            # A project line typically starts with an uppercase letter, has a pipe '|' or dash with spaces,
            # includes tech or date/github hints, and does NOT end with a period.
            is_proj_start = bool(
                re.match(r"^[A-Z][A-Za-z0-9\s\u2013\u2014\-\(\)\.,]+?\s*(?:\||\s+[\u2013\u2014]\s+)", line)
                and not line.strip().endswith(".")
                and any(k in line.lower() for k in [
                    "github", "gitlab", "202", "201", "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec", "|"
                ])
                and not line.startswith(("•", "-", "*", "1.", "2.", "3.", "4.", "5."))
            )

            if is_proj_start:
                flush_project()
                current_project_name = line
                current_project_lines = []
            else:
                if current_project_name is not None:
                    current_project_lines.append(line)
                else:
                    current_section_lines.append(line)
        else:
            current_section_lines.append(line)

    # Flush remaining
    if in_projects:
        flush_project()
    else:
        flush_section()

    return documents


# ── 2. Academic Transcript Parser ─────────────────────────────────────────────
def parse_academic(pdf_path: str) -> List[Document]:
    """
    Parse an academic transcript / marks card into structured Documents.
    Preserves key-value student info, course grade tables, and SGPA/CGPA summaries.
    """
    import pdfplumber

    stem = _clean_stem(pdf_path)
    documents: List[Document] = []
    chunk_counter = 0

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):
            text = page.extract_text() or ""
            lines = text.split("\n")

            kv_pairs: List[str] = []
            summary_lines: List[str] = []
            other_lines: List[str] = []

            for line in lines:
                s = line.strip()
                if not s:
                    continue
                # Key-value student metadata
                match = KV_LINE_RE.match(s)
                if match:
                    k, v = match.group(1).strip(), match.group(2).strip()
                    if len(k) < 40 and not any(w in k.lower() for w in ("marks", "min", "max", "sl", "course", "http")):
                        kv_pairs.append(f"{k}: {v}")
                        continue

                # SGPA / CGPA / Credits summary
                if any(w in s.lower() for w in ("sgpa", "cgpa", "total credits", "weighted percentage")):
                    summary_lines.append(s)
                    continue

                other_lines.append(s)

            # 1. Header Metadata Chunk (Registration, Student Name, Semester, Batch)
            if kv_pairs:
                chunk_counter += 1
                header_content = "Academic Record / Student Details:\n" + "\n".join(kv_pairs)
                documents.append(Document(
                    page_content=header_content,
                    metadata={
                        "source": pdf_path,
                        "filename": os.path.basename(pdf_path),
                        "page": page_num,
                        "document_type": "academic",
                        "section": "student_info",
                        "subsection": "header",
                        "item_type": "academic_header",
                        "item_name": "Student & Program Details",
                        "chunk_id": f"{stem}_header_{chunk_counter:03d}",
                    }
                ))

            # 2. Table rows extraction
            tables = page.extract_tables()
            table_rows_text: List[str] = []
            for table in tables:
                if not table:
                    continue
                for row in table:
                    if not row:
                        continue
                    cells = [c.strip().replace("\n", " ") for c in row if c and c.strip()]
                    if len(cells) >= 2:
                        table_rows_text.append(" | ".join(cells))
                    elif len(cells) == 1:
                        table_rows_text.append(cells[0])

            # 3. Course Grades Chunk
            combined_body = ""
            if table_rows_text:
                combined_body = "Courses and Grades Table:\n" + "\n".join(table_rows_text)
            elif other_lines:
                combined_body = "\n".join(other_lines)

            if combined_body:
                chunk_counter += 1
                documents.append(Document(
                    page_content=combined_body,
                    metadata={
                        "source": pdf_path,
                        "filename": os.path.basename(pdf_path),
                        "page": page_num,
                        "document_type": "academic",
                        "section": "course_grades",
                        "subsection": "grades",
                        "item_type": "academic_grades",
                        "item_name": f"Semester Grades (Page {page_num})",
                        "chunk_id": f"{stem}_grades_{chunk_counter:03d}",
                    }
                ))

            # 4. GPA Summary Chunk
            if summary_lines:
                chunk_counter += 1
                gpa_content = "Semester GPA & Academic Performance Summary:\n" + "\n".join(summary_lines)
                documents.append(Document(
                    page_content=gpa_content,
                    metadata={
                        "source": pdf_path,
                        "filename": os.path.basename(pdf_path),
                        "page": page_num,
                        "document_type": "academic",
                        "section": "academic_performance",
                        "subsection": "gpa_summary",
                        "item_type": "gpa_summary",
                        "item_name": "SGPA / CGPA Performance Summary",
                        "chunk_id": f"{stem}_gpa_{chunk_counter:03d}",
                    }
                ))

    return documents


# ── 3. Research Paper Parser ──────────────────────────────────────────────────
def parse_research_paper(pdf_path: str) -> List[Document]:
    """
    Parse a research paper PDF into structure-aware section chunks.
    Detects Abstract, Introduction, System Architecture, FPGA/Hardware Implementation,
    Evaluation, Conclusion, etc.
    """
    import pdfplumber
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    stem = _clean_stem(pdf_path)
    documents: List[Document] = []
    chunk_counter = 0

    section_heading_re = re.compile(
        r"^(?:(?:[I|V|X]+\.|\d+\.|\b[A-Z]\.)\s+)?(ABSTRACT|INTRODUCTION|BACKGROUND|RELATED WORK|SYSTEM ARCHITECTURE|SYSTEM DESIGN|METHODOLOGY|IMPLEMENTATION|SIMULATION|EVALUATION|RESULTS|EXPERIMENTAL RESULTS|CONCLUSION|REFERENCES)(?:.*)$",
        re.IGNORECASE
    )

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):
            text = page.extract_text() or ""
            if not text.strip():
                continue

            # Split into reasonable logical chunks
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=600,
                chunk_overlap=100,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            chunks = splitter.split_text(text)

            current_detected_section = "paper_content"
            for chunk_str in chunks:
                # Check if chunk starts with or contains a section header
                for line in chunk_str.split("\n")[:3]:
                    m = section_heading_re.match(line.strip())
                    if m:
                        current_detected_section = m.group(1).lower().replace(" ", "_")
                        break

                chunk_counter += 1
                chunk_id = f"{stem}_p{page_num}_{chunk_counter:03d}"
                documents.append(Document(
                    page_content=chunk_str.strip(),
                    metadata={
                        "source": pdf_path,
                        "filename": os.path.basename(pdf_path),
                        "page": page_num,
                        "document_type": "research_paper",
                        "section": current_detected_section,
                        "subsection": None,
                        "item_type": "paper_section",
                        "item_name": current_detected_section.replace("_", " ").title(),
                        "chunk_id": chunk_id,
                    }
                ))

    return documents


# ── 4. General / Other Document Parser ────────────────────────────────────────
def parse_other_document(file_path: str) -> List[Document]:
    """
    Universal Parser for ANY arbitrary PDF, DOCX, TXT, Markdown, or general document.
    Reliably extracts text and tables page-by-page without dropping content or corrupting binary files.
    """
    stem = _clean_stem(file_path)
    documents: List[Document] = []
    ext = os.path.splitext(file_path)[1].lower()
    filename = os.path.basename(file_path)

    from langchain_text_splitters import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=120)

    if ext == ".pdf":
        import pdfplumber
        chunk_counter = 0
        try:
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    text = page.extract_text() or ""
                    # Also extract any tables from the page
                    tables = page.extract_tables() or []
                    if tables:
                        table_texts = []
                        for tbl in tables:
                            if not tbl:
                                continue
                            tbl_rows = [" | ".join(str(cell or "").strip() for cell in row) for row in tbl if any(row)]
                            if tbl_rows:
                                table_texts.append("\n".join(tbl_rows))
                        if table_texts:
                            text += "\n\nTables:\n" + "\n\n".join(table_texts)

                    if not text.strip():
                        continue

                    chunks = splitter.split_text(text)
                    for chunk_str in chunks:
                        chunk_counter += 1
                        chunk_id = f"{stem}_p{page_num}_{chunk_counter:03d}"
                        first_line = chunk_str.strip().split("\n")[0][:60]
                        documents.append(Document(
                            page_content=chunk_str.strip(),
                            metadata={
                                "source": file_path,
                                "filename": filename,
                                "page": page_num,
                                "document_type": "other",
                                "section": "document_content",
                                "subsection": None,
                                "item_type": "document_chunk",
                                "item_name": first_line,
                                "chunk_id": chunk_id,
                                "text": chunk_str.strip(),
                            }
                        ))
        except Exception as e:
            print(f"[structure_parser] pdfplumber failed on {file_path}: {e}")

        # Fallback to pypdf if pdfplumber extracted nothing
        if not documents:
            try:
                import pypdf
                reader = pypdf.PdfReader(file_path)
                chunk_counter = 0
                for page_num, p in enumerate(reader.pages, 1):
                    p_text = p.extract_text() or ""
                    if not p_text.strip():
                        continue
                    chunks = splitter.split_text(p_text)
                    for chunk_str in chunks:
                        chunk_counter += 1
                        documents.append(Document(
                            page_content=chunk_str.strip(),
                            metadata={
                                "source": file_path,
                                "filename": filename,
                                "page": page_num,
                                "document_type": "other",
                                "section": "document_content",
                                "subsection": None,
                                "item_type": "document_chunk",
                                "item_name": filename,
                                "chunk_id": f"{stem}_p{page_num}_{chunk_counter:03d}",
                                "text": chunk_str.strip(),
                            }
                        ))
            except Exception as e:
                print(f"[structure_parser] pypdf fallback failed on {file_path}: {e}")

    elif ext == ".docx":
        try:
            import docx
            doc = docx.Document(file_path)
            parts = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
            for tbl in doc.tables:
                tbl_rows = [" | ".join(cell.text.strip() for cell in row.cells) for row in tbl.rows]
                if tbl_rows:
                    parts.append("\n".join(tbl_rows))
            content = "\n\n".join(parts)
            chunks = splitter.split_text(content)
            for i, chunk_str in enumerate(chunks, 1):
                chunk_id = f"{stem}_doc_{i:03d}"
                documents.append(Document(
                    page_content=chunk_str.strip(),
                    metadata={
                        "source": file_path,
                        "filename": filename,
                        "page": 1,
                        "document_type": "other",
                        "section": "document_content",
                        "subsection": None,
                        "item_type": "document_chunk",
                        "item_name": filename,
                        "chunk_id": chunk_id,
                        "text": chunk_str.strip(),
                    }
                ))
        except Exception as e:
            print(f"[structure_parser] Error reading docx {file_path}: {e}")

    else:
        # Plain text, Markdown, CSV, or general text files
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            chunks = splitter.split_text(content)
            for i, chunk_str in enumerate(chunks, 1):
                first_line = chunk_str.strip().split("\n")[0][:60].strip("# ")
                chunk_id = f"{stem}_sec_{i:03d}"
                documents.append(Document(
                    page_content=chunk_str.strip(),
                    metadata={
                        "source": file_path,
                        "filename": filename,
                        "page": 1,
                        "document_type": "other",
                        "section": re.sub(r"[^a-z0-9]+", "_", first_line.lower()).strip("_") or "content",
                        "subsection": None,
                        "item_type": "document_chunk",
                        "item_name": first_line,
                        "chunk_id": chunk_id,
                        "text": chunk_str.strip(),
                    }
                ))
        except Exception as e:
            print(f"[structure_parser] Error reading text file {file_path}: {e}")

    return documents


# ── 5. Certificate & Online Course Parser ──────────────────────────────────────
def parse_certificate(pdf_path: str) -> List[Document]:
    """
    Parse a course completion certificate (e.g. Coursera, Udemy, AWS) into
    metadata-rich Documents with section='certifications' and document_type='certificate'.
    """
    import pdfplumber
    stem = _clean_stem(pdf_path)
    filename = os.path.basename(pdf_path)
    documents: List[Document] = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):
            text = page.extract_text() or ""
            if not text.strip():
                continue

            # Detect course / credential title
            lines = [l.strip() for l in text.split("\n") if l.strip()]
            item_name = filename
            for l in lines:
                if any(w in l.lower() for w in ["generative ai", "machine learning", "deep learning", "python", "large language models", "specialization", "course"]):
                    if not any(stop in l.lower() for stop in ["authorized by", "offered through", "verify at", "this cert"]):
                        item_name = l
                        break

            chunk_id = f"{stem}_cert_p{page_num}"
            documents.append(Document(
                page_content=text.strip(),
                metadata={
                    "source": pdf_path,
                    "filename": filename,
                    "page": page_num,
                    "document_type": "certificate",
                    "section": "certifications",
                    "subsection": None,
                    "item_type": "certificate",
                    "item_name": item_name,
                    "chunk_id": chunk_id,
                    "text": text.strip(),
                }
            ))

    if not documents:
        # Fallback if no pages extracted
        documents.append(Document(
            page_content=f"Certificate document: {filename}",
            metadata={
                "source": pdf_path,
                "filename": filename,
                "page": 1,
                "document_type": "certificate",
                "section": "certifications",
                "subsection": None,
                "item_type": "certificate",
                "item_name": filename,
                "chunk_id": f"{stem}_cert_001",
                "text": f"Certificate document: {filename}",
            }
        ))

    return documents


# ── Unified Public Parser Entrypoint ──────────────────────────────────────────
def parse_document(file_path: str, file_hash: Optional[str] = None) -> List[Document]:
    """
    Master parser: classifies document by content, then delegates to specialized
    structure-aware parser.
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        import pdfplumber
        with pdfplumber.open(file_path) as pdf:
            sample_text = "\n".join(p.extract_text() or "" for p in pdf.pages[:3])
    elif ext == ".docx":
        try:
            import docx
            doc = docx.Document(file_path)
            sample_text = "\n".join(p.text for p in doc.paragraphs[:20] if p.text.strip())
        except Exception:
            sample_text = ""
    else:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            sample_text = f.read(4000)

    doc_type = classify_document_content(sample_text, os.path.basename(file_path))

    if doc_type == "resume":
        chunks = parse_resume(file_path)
    elif doc_type == "academic":
        chunks = parse_academic(file_path)
    elif doc_type == "certificate":
        chunks = parse_certificate(file_path)
    elif doc_type == "research_paper":
        chunks = parse_research_paper(file_path)
    else:
        if ext == ".pdf":
            # Only treat as research paper if research signals are actually present
            research_signals = ["abstract", "ieee", "proceedings", "conference", "doi", "references"]
            if any(sig in sample_text.lower() for sig in research_signals):
                chunks = parse_research_paper(file_path)
            elif any(w in sample_text.lower() for w in ["certi", "coursera", "udemy", "course", "completed"]):
                chunks = parse_certificate(file_path)
            else:
                chunks = parse_other_document(file_path)
        else:
            chunks = parse_other_document(file_path)

    # Ensure text and file_hash are present on all chunks
    for c in chunks:
        if "text" not in c.metadata:
            c.metadata["text"] = c.page_content
        if file_hash:
            c.metadata["file_hash"] = file_hash

    return chunks



# ── Phase 5: Section Index Builder ────────────────────────────────────────────
def build_section_index(documents: List[Document]) -> Dict[str, Any]:
    """
    Build dynamic section-level index for all ingested documents.
    Specifically captures items under structured sections like 'projects', 'education', etc.
    """
    index: Dict[str, Any] = {}

    for doc in documents:
        meta = doc.metadata
        fn = meta.get("filename") or os.path.basename(meta.get("source", "unknown"))
        doc_type = meta.get("document_type", "other")
        sec = meta.get("section", "general")
        item_type = meta.get("item_type", "")
        item_name = meta.get("item_name", "")
        chunk_id = meta.get("chunk_id", "")

        if fn not in index:
            index[fn] = {
                "document_type": doc_type,
                "source": meta.get("source", ""),
                "sections": {}
            }

        sections = index[fn]["sections"]
        if sec not in sections:
            sections[sec] = {
                "chunk_ids": [],
                "items": [],
                "item_types": []
            }

        if chunk_id and chunk_id not in sections[sec]["chunk_ids"]:
            sections[sec]["chunk_ids"].append(chunk_id)

        if item_name and item_name not in sections[sec]["items"]:
            sections[sec]["items"].append(item_name)
            sections[sec]["item_types"].append(item_type)

    return index
